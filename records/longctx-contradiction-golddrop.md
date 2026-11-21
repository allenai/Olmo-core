---
title: Gold-document drop vs dense LoRA on contradiction at 256k and 1M (Beaker, FLOP-matched, held-out CE)
date: 2026-11-19
updated: 2026-11-20
experiment: tokdrop_lr
area: token_routing
kind: result
status: current
superseded_by: 
artifact: https://claude.ai/artifact/5TcD7ho7SMhChK7jeJgi6M
summary: At matched training FLOPs the recipe + gold-document drop beats dense LoRA on held-out contradiction CE at both lengths. 256k (15.2k PF, eval_size 160): 0.029 vs 0.092 at gold drop's final 14.4k PF (-0.063 +- 0.011 paired), dense's final CE reached at ~7.6k PF (2.0x; 2.2-4.1x at intermediate levels). 1M (13.6k PF): 1M-held-out CE 0.239 vs 1.393 for dense LoRA on 84 1M rows (>=5.2x FLOPs-to-match) and vs 1.397 for dense LoRA on the same data mix as gold drop. Dense LoRA on 1M rows alone gets worse at 256k/16k. eval_size 64-160 per set (below 500).
---
# Gold-document drop vs dense LoRA on contradiction at 256k and 1M

Prasann (2026-11-18/19): contradiction-only training at 256k and 1M on Beaker, gold-document drop on top of the recipe
vs dense LoRA, each sized by what dense LoRA trains in 32 GPU-hours, the gold-drop arm FLOP-matched (more rows).
Code and launch commands: `research/token_routing/early_loss_proxy/longctx_contra/` (README there) on branch
`claude/longctx-contra-golddrop`. Numbers: teacher-forced answer CE on held-out rows, no-cot, logged with
`ops result add` (experiment `tokdrop_lr`, metric `heldout_answer_ce`).

## Verdict

**Gold drop wins at matched FLOPs at both lengths, and the gap does not close by the end of the budget.**

- **256k** (dense 15,203 PF; gold drop ended its epoch at 14,421 PF = 95% of dense):
  - lc256k held-out CE at gold drop's final PF: **0.029 vs 0.092** (dense interpolated), paired diff -0.063 +- 0.011.
  - FLOPs to match dense's final CE (0.088): gold drop reaches it at ~7,630 PF, **2.0x**. At dense's intermediate
    checkpoints: 2.2x (3,682-PF level), 3.2x (5,530), 4.1x (7,407), 2.8x (9,259, 11,116), 2.2x (12,962).
  - Matched-PF paired differences (gold drop - dense, lc256k): -0.311 / -0.190 / -0.144 / -0.114 / -0.074 / -0.079 /
    -0.076 at dense's 1.8k / 3.7k / 5.5k / 7.4k / 9.3k / 11.1k / 13.0k PF (SE 0.010-0.015).
  - 16k held-out (short-row skill): gold drop 0.0002 vs dense 0.028 at 14,421 PF (-0.028 +- 0.004); gold drop reaches
    dense's final 16k CE at ~3,340 PF (4.6x). Confounded: gold drop trained on 72k short rows vs dense's 4,800.
- **1M** (dense LoRA 13,578 PF on 84 1M rows only; gold drop 14,696 PF, full epoch):
  - lc1m held-out CE: dense 1.699 -> **1.393**; gold drop **0.239** at the matched 13,578 PF (paired -1.154 +- 0.030,
    64 rows). Gold drop's first lc1m-scored checkpoint (2,620 PF, 0.698) is already below dense's final, so
    FLOPs-to-match >= 5.2x (no lc1m score was taken before 2,620 PF).
  - Dense LoRA on 1M rows alone does not improve 256k (1.566 -> 1.558) and degrades 16k (0.977 -> 1.414); gold drop
    reaches 0.078 / 0.015 there.
  - Dense LoRA on the SAME data mix as gold drop (`dense1m_mix`, 83 1M rows + 1,432 short rows, 13,620 PF):
    lc1m 1.699 -> **1.397** at 13,620 PF (gold drop at matched PF 0.239, paired -1.159 +- 0.031; >= 5.2x), and again no
    gain at 256k (1.512) and a loss at 16k (1.290) despite ~1,350 effective short rows. The short rows do not rescue
    dense LoRA at 1M.
  - Why dense LoRA trained at 1M damages 16k/256k (while dense LoRA at 256k does not) - **the CP8 training path is ruled
    out** (2026-11-20): step-1 LoRA gradients from dense256b's step-75 adapter agree between CP4 and CP8, with and without
    the 1M memory flags (int32 guard, GDN segments 131k, 16k FFN chunks): global cosine 0.99993 / 0.99992 on a 256k row,
    0.99994 on a packed short window, every block >= 0.9998, relative norm difference 1.1-1.3% (bf16), identical losses;
    and 75 steps of dense256b's exact config under CP8 + those flags score lc256k 0.428 / lc16k 0.170 vs dense256b's
    0.507 / 0.197 at the same 1,827.5 PF (paired -0.080 +- 0.013 / -0.027 +- 0.011: CP8 is not worse). Scoring and adapter
    loading are identical for all arms (1139 keys loaded), and the LR schedule only under-trains (6x lower cumulative LR
    at matched PF, 84-93 optimizer steps). What remains is the training data itself: 84-93 single-row steps on
    1M-token rows whose positions run far past the native 262k RoPE range (no YaRN), which plausibly degrades shorter
    contexts; not tested further (a YaRN or 512k dense run would).

## Setup

- Model: Qwen3.5-4B base (`q35-4b-base-markerfix`), LoRA rank 64 / scale 32 on every block (dense arms: `--lora-rank 64`;
  router arms: the recipe's early LoRA, same parameter names), LR 1e-5, schedule over each arm's own steps.
- Gold drop = the recipe router (`--st-otr-stages 7:0.2,16:0.07`) + whole-document drop before block 0
  (`src/olmo_core/nn/gold_doc_drop.py`): every gold claim kept plus a random share r ~ U(lo, hi) of the distractors;
  a share of rows kept whole.
- Data: ctc `contradiction` (PubMed claim pairs; 3 gold pairs = 6 gold claims per row; fillers from 185k sentences).
  At most ~18.1k rows with distinct gold per length. Rows: 256k = 5,550 claims (mean 252k tokens, max 261,891);
  1M = 22,500 claims (mean 1.039M). Short rows: 2k/4k/8k/16k cells as keepfrac_len's kl pools.
- Held-out (`heldout_lc.npz`, ctc eval split, eval seed 99, gold disjoint from training): lc256k 160 rows,
  lc1m 64 rows, lc16k 160 rows. ⚠ all below 500: binomial-style SE of a mean CE here is ~0.01-0.03; paired
  differences use the same rows in both arms.
- Scoring: checkpoint-then-score with one forward per set for every arm, all in the dense-LoRA model class: 256k/16k
  on the 1-GPU logits path (the reporting scale of earlier 16k-64k results); 1M under Ulysses CP8 (labels path with the
  z-loss removed; agrees with the 1-GPU path to -0.0035 mean on 256k rows, CP4 vs CP8 base 1.698 / 1.699).
- FLOPs: the trainer's FLOP meter (actual PF, attention charged causal-quadratically; corrected here to count each
  row once under CP). Matched-PF values interpolate the other arm per row between its neighbouring checkpoints.

## The arms

| arm | Beaker | data | config | PF | wall |
|---|---|---|---|---|---|
| dense LoRA 256k (`dense256b`) | 01M7SJP9PY8W3H8B2JM7SXW87S | 1,091 x 256k + 1,200 each at 2k/4k/8k/16k = 5,891 rows | 8 H100, CP4 x DP2, packed (doc lengths), 2 windows of 262k per step, 615 steps | 15,203 | 4.3 h |
| gold drop 256k (`gdd256`) | 01M7SHSBVSZ7HR6KQZ83V6HYW0 | 18,000 distinct-gold + 1,980 reused-gold 256k rows (final stretch, from about 13.7k PF) + 71,993 short = 91,973 rows | 8 H100 DP8, keep 0.05-0.5, 5% whole, prefix-chunk 4096, 8 rows/step in length-homogeneous blocks, 11,496 steps | 14,421 | 11.5 h |
| dense LoRA 1M, 1M rows only (`dense1m`) | 01M7SQ9K76E66V5R1KV24B68AB | the first 84 1M rows of gdd1m's order | 8 H100 CP8, `--gdn-int32-safe --ffn-chunk-tokens 32768`, 1 row/step, 84 steps | 13,578 | 3.9 h |
| dense LoRA 1M, gold drop's mix (`dense1m_mix`) | 01M7TS4XFVTKKGS5HTD47QXQVS | the first 1,515 rows of gdd1m's order: 83 1M + 1,432 short (short:long 17.3 vs 14.5) | CP8, packed 1M windows; windows starting with a 1M row run unpacked (co-packed docs masked, ~80 short rows lost), short windows packed with GDN calls grouped at 131k; 93 steps | 13,620 | 4.05 h |
| gold drop 1M (`gdd1m`) | 01M7SKA30E465Z6AZQTDZEJEEA | 4,971 1M rows + 71,993 short = 76,964 rows | 8 H100 DP8, keep 0.05-0.4, 0% whole, prefix-chunk 4096, `--gdn-int32-safe`, 9,620 steps | 14,696 | 10.3 h |

How the data differ at matched FLOPs (the point of FLOP matching): at 256k gold drop trains on ~18x the long rows and
~15x the short rows of dense (91,973 vs 5,891 rows); at 1M ~60x the 1M rows (4,971 vs 83-84) and ~50x the short rows
of `dense1m_mix` (dense1m has none). Gold drop sees each row with most distractors removed (kept share ~5-50%).

## Tables (CE vs training PF; eval_size flagged)

lc256k (160 rows ⚠):

| PF | dense256b | gdd256 |
|---|---|---|
| 0 | 1.566 | 1.566 |
| ~1.8k | 0.507 | 0.196 |
| ~3.7k | 0.326 | 0.133 |
| ~5.5k | 0.254 | 0.109 |
| ~7.4k | 0.204 | 0.089 |
| ~9.2k | 0.147 | 0.074 |
| ~11k | 0.130 | 0.054 |
| ~12.8k | 0.105 | 0.030 |
| 14.4k-15.2k | 0.088 (14.8k), 0.088 (15.2k) | 0.029 (14.4k) |

1M arms (lc1m 64 rows ⚠ under CP8; lc256k / lc16k 160 rows ⚠):

| PF | dense1m lc1m / lc256k / lc16k | gdd1m lc1m / lc256k / lc16k |
|---|---|---|
| 0 | 1.699 / 1.566 / 0.977 | same |
| 2.6k | | 0.698 / 0.288 / 0.053 |
| 3.4k-3.6k | 1.466 / 1.658 / 1.325 (3,391) | 0.543 / 0.225 / 0.042 (3,592) |
| 6.4k-6.8k | 1.418 / 1.523 / 1.442 (6,793) | 0.638 / 0.222 / 0.038 (6,420); 0.391 / 0.149 / 0.020 (7,511) |
| 10.0k-10.2k | 1.396 / 1.613 / 1.536 (10,188) | 0.314 / 0.119 / 0.024 (10,027) |
| 12.8k-13.8k | 1.393 / 1.558 / 1.414 (13,578) | 0.252 / 0.081 / 0.012 (12,785); 0.236 / 0.083 / 0.012 (13,810) |
| 14.7k | | 0.239 / 0.078 / 0.015 |

dense1m_mix (same eval sets): 0 PF 1.699 / 1.566 / 0.977; 3,909 PF 1.420 / 1.565 / 1.389; 7,792 PF 1.448 / 1.661 / 1.582;
11,655 PF 1.405 / 1.593 / 1.545; 13,620 PF 1.397 / 1.512 / 1.290 (lc1m / lc256k / lc16k). Matched-PF paired differences
vs gold drop on lc1m: -0.866 / -1.066 / -1.114 / -1.159 (SE 0.029-0.032).

## Caveats

- **eval_size 64-160** per held-out set, below the 500 default. Paired per-row differences use the same rows in both
  arms; checkpoint-to-checkpoint wobble in the gold-drop curves is 0.03-0.1 (e.g. gdd1m lc1m 0.543 at 3.6k PF, 0.638 at
  6.4k, 0.391 at 7.5k), so single checkpoints are noisy; the conclusions rest on gaps of 0.06-1.5 CE.
- **dense1m has no short rows** (1M rows only; the "1M rows only" reference). `dense1m_mix` uses gold drop's mix but
  ~80 of its 1,432 short rows sit after a 1M row in a packed window and are masked out, and its short:long row ratio
  (17.3) is a little higher than gdd1m's (14.5).
- **gdd1m used 0% whole rows and keep_hi 0.4** (1M whole rows and keeps of 0.5 do not fit on an H100); gdd256 used 5%
  whole rows. gdd256 stopped at its epoch end at 95% of dense's PF (no top-up; compared at matched PF up to there).
- **Positions past 262,144** are native RoPE extrapolation (the trainer has no YaRN for Qwen3.5): gold-drop 1M rows
  keep up to ~430k tokens with positions up to the kept length; dense 1M training and every 1M score extrapolate to
  1.04M.
- **Short-row comparisons are data-confounded**: gold drop trains on 15-50x more short rows at matched FLOPs.
- **FLOPs, not wall-clock**: at matched FLOPs gdd256 took 11.5 h vs 4.3 h on 8 H100s (short kept rows, prefix-chunk
  overhead, data-parallel stragglers reduced by length-homogeneous steps).
- The gold-drop arms trained in file order with length-homogeneous 8-row steps (all-long or all-short), shuffled at the
  block level; the dense arms packed short rows into windows. Loss is normalised per answer token over the global batch
  in all arms.
- gdd256's last ~10% of FLOPs (from ~13.7k PF) reuse the gold pairs of 1,980 earlier rows with new fillers; every
  matched point up to 12.7k PF is distinct-gold only.

## Findings along the way (all on `claude/longctx-contra-golddrop` unless noted)

- **Gated DeltaNet int32 limit**: fla 0.4.1's `chunk_gated_delta_rule` overflows its per-chunk state index from
  2^30 / (H_v x 128) tokens (262,144 for Qwen3.5-4B), silently corrupting outputs past that token ("dense forward breaks
  past ~262k"). Confirmed at kernel level and fixed (`--gdn-int32-safe`, exact segmented calls) on
  `claude/dense-1m-pipeline` (commit a5a6d7ec4, its README `research/token_routing/dense_longctx/README.md`). Ulysses CP
  scales the bound by the CP degree; `--prefix-chunk` keeps the router arms' blocks 0-6 at 4,096-token calls. Extended
  here to packed rows (per-group calls / state-carrying segments; max abs diff 0.0 vs the plain kernel,
  `test_packed_guard.py`).
- **Hidden z-loss**: olmo_core's LM head passes `z_loss_multiplier or 1e-4` to liger's fused-linear CE, which always adds
  1e-4 * lse^2 to the loss. Logged train/CE (and any labels-path CE) is ~0.05-0.09 per token higher than plain CE, and
  every fused-CE training run has carried a small z-loss regulariser. Same scored tokens as the logits path. Not changed
  in training; the scoring path removes it.
- **`--pack` emitted no document lengths**: packed examples attended to each other and the FLOP meter charged one
  example per window. Fixed (`generate_doc_lengths`). The FLOP meter also counted every CP rank; fixed.
- Beaker node jupiter-cs-aus-152 (and at least one other) fails container start with a missing weka mount;
  `launch_lc_smoke.py --exclude-host` pins jobs elsewhere.

## Where things are

Code / launchers / analysis: `research/token_routing/early_loss_proxy/longctx_contra/` (`launch_arms.sh`,
`analyze_lc.py`, `record_lc.py`, `assemble_mix.py`, `build_lc_pools.py`). Pools: mooney `/data/prasann/early_loss_proxy/shards/`
and weka `.../prasanns/early_loss_proxy/shards/` (`lc256kmix_dense`, `lc256kmix_gdd`, `lc1mdense84`, `lc1mmixdense`,
`lc1mmix_gdd`, `heldout_lc.npz`). Checkpoints: weka `.../early_loss_proxy/runs_lc/<arm>/stepN`. Score and run JSONs:
S3 `early_loss_proxy/results/lcA-*` and `lcS-*`.

## Group-partitioned attention on top of gold drop (2026-11-20/21, branch `claude/attn-groups-256k`)

Prasann asked for this. In the attention layers only (GDN layers and FFNs keep the full post-drop row), each training row's
documents are split every forward into **hubs** (the 6 gold claims + 6 random decoy distractors + the preamble) and
**G = 4 random groups** of the rest. A group token attends to its group + hubs, a hub token to hubs only, and the
question/answer strip to everything. A share `full_frac` of rows keeps full attention. Every set runs as its own
gathered flash-attn varlen sequence (a real compute saving, also inside `--prefix-chunk` and after the router's
mid-stack drops). The FLOP meter charges every pair actually computed. Scoring is unchanged (full attention, same
scorer and rows). Code: `src/olmo_core/nn/attention/group_attention.py`, flags `--st-attn-groups /
--st-attn-group-decoys / --st-attn-group-full-frac / --st-attn-group-symmetric`, runner and analysis in
`research/token_routing/early_loss_proxy/longctx_contra/attn_groups/` (README there), arms `launch_arms.sh gddag`.

**Verdict: group attention hurts held-out CE badly at a modest saving; not worth continuing in this form.**
- At G = 4 with full_frac 0.1, lc256k CE at ~1.82k PF is **0.531 vs gdd256's 0.199** (paired +0.332 +- 0.020,
  eval_size 160 ⚠). That is no better than dense LoRA (0.513). Training CE on the same rows is *lower* than gdd256's
  (0.025 vs 0.039, steps 1174-1574): the model fits the grouped task, and that does not carry over to full attention.
  We stopped the arm at ~1.9k PF.
- The symmetric-hub fix (hub documents also join a random group, so every document sees "own group + hubs") recovers
  only part of the gap: **0.435** (+0.238 +- 0.017 vs gdd256). The small-context hub cue explains some of the gap; the
  rest is the grouped-vs-full mismatch itself.
- With full_frac 0.5 the gap shrinks to **0.251** (+0.055 +- 0.013, inside the 0.08 run-to-run noise), but the
  per-step saving is then only 9%.
- Per-step saving vs gdd256, measured on Beaker 8xH100 over identical batches (batch hashes match on every step):
  PF ratio 0.830 / time ratio 0.822 (G = 4, full_frac 0.1, either hub mode); 0.912 / 0.918 at full_frac 0.5. Peak
  memory 37.0 vs 35.8 GiB (no saving: the peak is not in attention).

| arm (first checkpoint) | PF | lc256k CE ⚠160 | vs gdd256 (paired) | vs dense256b | lc16k CE ⚠160 | vs gdd256 | train CE 1174-1574 |
|---|---|---|---|---|---|---|---|
| gdd256 (reference, interpolated) | ~1,820 | 0.199 | | | 0.042 | | 0.039 |
| dense256b (reference, interpolated) | ~1,820 | 0.513 | | | 0.201 | | |
| `gddag4` G4, ff 0.1 (step 1690) | 1,818 | 0.531 | +0.332 +- 0.020 | +0.018 +- 0.021 | 0.185 | +0.143 +- 0.014 | 0.025 |
| `gddag4sym` symmetric hubs (step 1690) | 1,820 | 0.435 | +0.238 +- 0.017 | -0.077 +- 0.021 | 0.101 | +0.060 +- 0.010 | 0.033 |
| `gddag4ff05` ff 0.5 (step 1574) | 1,828 | 0.251 | +0.055 +- 0.013 | -0.257 +- 0.015 | 0.056 | +0.016 +- 0.008 | 0.038 |

Matched rows (the same optimizer step as gdd256, so the same rows seen): +0.345 / +0.249 / +0.060 on lc256k, i.e. the
gap is a per-row loss, not a FLOP-matching artefact. None of the arms reaches gdd256's CE at its first checkpoint.

Screen (1 H200, 12 single-row steps on real lc256kmix_gdd 256k rows, full_frac 0, paired vs attention off): on
gold-dropped rows G = 2/4/8 cut per-step PF to 0.886 / 0.828 / 0.800 of off, and wall-clock to 0.97 / 0.96 / 0.95.
On whole 256k rows PF falls to 0.771 / 0.656 / 0.599 and wall-clock to 0.777 / 0.654 / 0.598. Peak memory is
unchanged. G = 1 costs the same as off.

Tests (`src/test/nn/attention/group_attention_test.py`, GPU): G = 1 with no hubs matches full flash attention (max
abs output diff 3.9e-3 in bf16; exactly 0 through the Attention module). For G = 2/4/8, both hub modes, outputs match
an fp32 masked-SDPA reference to the same 7-9e-3 error as flash full attention; relative grad error is at most 1.1%.
Chunked-prefix vs whole row: 0. Meter pair counts match hand counts.

Beaker: gddag4 01M7XSSH3PHNMYBB4P9819XK9E (stopped), gddag4ff05 01M7XZZ3MCYAYZ3KSGCPJ53BEF, gddag4sym
01M7Y02KN9AHQK7JEK6DHHKANH; scores 01M7XZ70Z942VJWSKNF9MH5413 / 01M7Y5FA4NM2AWQ2VEN99FV195 /
01M7Y5GFEAC30YFMHV156W73XF. Checkpoints: weka `runs_lc/gddag4{,ff05,sym}/stepN`. About 44 GPU-h in total.
