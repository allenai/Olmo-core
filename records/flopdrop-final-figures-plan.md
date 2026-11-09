---
title: Plan — CPT and setA-SFT experiment sets for the final FLOP-dropping figures (A1 FLOPs-to-α, f1-vs-FLOPs ladders, A2 training memory)
date: 2026-10-16
updated: 2026-10-16
experiment:
area: token_routing
kind: plan
status: current
superseded_by:
summary: Proposed experiment sets (continued pretraining; setA SFT + CTC eval) that produce the three final figures, preceded by a held-out-dev tuning phase that fixes the token-drop + layer-exit method.
---
# Plan — experiment sets for the final FLOP-dropping figures

Nothing here is launched. Section 9 lists the decisions Prasann needs to make before Phase T.

## 0. The three target figures and what each needs

| figure | y | x | curves | raw data it needs |
|---|---|---|---|---|
| **A1** | training FLOPs to reach accuracy α | context length | {high-CTC, low-CTC} × {naive dense, our method} | for every (length, class, arm), a FLOP ladder that **brackets** α, so the crossing is interpolated, never extrapolated |
| **f1 vs FLOPs** (mock-up 2) | class-mean f1 | training PF (log) | one line per arm / compression level; one panel per (length, class) | the same ladders, plotted directly. A1 is read off these panels |
| **A2** | peak training memory | context length | standard vs our method at the FLOP-optimal setting, ≤2× and ≤4× its FLOPs | (i) FLOPs-to-α for several compression levels at the same length; (ii) measured peak memory for each level and length |

So one design serves all three figures: **ladders over (length × class × arm × budget)**, plus a cheap
memory benchmark. A2 additionally needs **more than one compression level** of our method at some lengths.

## 1. "Our method": what is best today, and why it needs a tuning pass first

Evidence on the candidates (all from the experiment pages):

| candidate | evidence | status as a *training* method |
|---|---|---|
| Input-level token drop with a pooled slot (ds64 soft-token: gold docs whole + first-K / first_last of the rest) | 2.2-4.0× fewer training FLOPs at f1 parity on contradiction, outlier, nq and oolong (5 rungs × 500) | best trained evidence. But it ran per task on unpacked rows, so it does **not** run on packed setA. Every arm cost 1.6-9.7× **more** GPU-hours (no compact-then-pack) |
| Token exit at layer L (dropstruct: dropped tokens leave the sequence from block L on) | L8 halves the random-drop penalty after CPT. Full-context CE +0.013 to +0.022 ⚠ eval_size=96 | the natural **token-drop × layer-skip** combination: dropped tokens skip layers ≥ L. Run only as CPT at 8k |
| Frozen keep rules (devloss_grid `gold_fl20p8` + skip odd layers) | 13/17 tasks within 0.05 nats at 2k/8k; skipping odd layers ≈ free at inference; GDN layers are **not** skippable ⚠ eval_size 8-16 | a screen only: frozen screens rank inference usability, not training value |
| Learned token + per-(token, layer) skip router (learned_router) | 4.48× mean per-task FLOP cut at ≤0.05 nats at 2k, but 2k routers lose most 32k cells, and layer keep 0.5 breaks at 32k ⚠ eval_size ≤64 | inference-only on frozen models; does not transfer across length |
| Random null-FFN drop (ffndrop) | below the dense frontier; confounded by the CPT data | not a candidate for the main figures |

**Verdict.** No single configuration is validated as a trained, packed, generation-graded method at
32k+. The best-supported family is **rule-based token dropping in which dropped tokens run only the
first L_exit layers** ("TX"). With L_exit = 0 this is ds64-style input drop. With L_exit > 0 it is the
token-drop + layer-skip combination. The plan therefore starts with a tuning phase on a held-out dev set
(Phase T) and freezes one config before the main grids.

### Method family TX (the search space)

**Rule (Prasann, 2026-10-16): one end-to-end approach for every task.** No per-task keep rules, budgets
or knobs. Gold labels may be used by the SFT keep rule. Any tuning is of global knobs only, selected
on a score pooled across tasks. The dev-loss probe may be used to check that the approach is viable.

**Keep rule = `gold_fl20p8_noslot`**, applied identically to every task: gold documents whole + an
8-token real prefix per document + `first_last` 20% of every other document, with no pooled slot. The
slot changes nothing: `gold_fl20p8` with the `cent_cmean` slot is within ±0.01 of noslot in every cell
below. This is the devloss grid's leading task-general rule and the learned-router "bar".

Per-task ΔCE vs full context for the class tasks. Source: the Dev-Loss Scheme Grid artifact
(claude.ai/code/artifact/5f22d13d-544b-4907-b31d-6be6afb537af). Frozen per-task dense 4B suite
checkpoints, teacher-forced answer CE, paired ± SE. ⚠ 8-48 rows per cell. The grid CSV itself is no
longer on disk.

| task | 2k | 8k | 32k | compaction @32k |
|---|---|---|---|---|
| nq | −0.002 ± 0.001 | −0.036 ± 0.030 | −0.006 ± 0.004 | ×0.25 |
| oolong | +0.077 ± 0.054 | +0.027 ± 0.030 | +0.000 ± 0.000 | ×0.31 |
| **rerank** | +0.031 ± 0.024 | −0.066 ± 0.009 | **+0.121 ± 0.014** | ×0.32 |
| contradiction | +0.000 ± 0.001 | +0.000 ± 0.001 | −0.032 ± 0.024 | ×0.35 |
| outlier | −0.034 ± 0.033 | −0.036 ± 0.038 | −0.040 ± 0.064 | ×0.26 |
| qdmatch_hpqa (proxy for qdmatch_nq) | −0.000 | −0.009 ± 0.009 | −0.009 ± 0.009 | ×0.32 |
| grouping / strmatch / xabsence (E9 only) | +0.021 / 0.000 / +0.019 | −0.002 / 0.000 / +0.012 ± 0.103 | +0.012 / 0.000 / −0.119 ± 0.046 | ×0.27-0.44 |
| median over all 17 grid tasks | +0.002 | −0.000 | +0.000 | ×0.31 (mean) |

Reading:
- The rule is viable on every class task except **rerank at 32k** (+0.12; it ranks 365 mostly non-gold passages).
- hotpotqa and qdmatch_nq were dropped from the grid and have **no cells**. T1 must cover them.
- **Uniform odd-layer skip is NOT free per task.** `gold_fl20p8_skipodd` puts rerank at +0.55 at 32k,
  contradiction at +0.07 and xabsence/outlier at −0.2. The "≈ free" median hid this. Layer skipping
  therefore has to be per-token (token exit), not a uniform skip of whole blocks.
- Outlier's negative ΔCE under gold-forced keep is the shortcut signature: the uncompressed documents
  are the answer (see q below).

For each document in the context (SFT: each `<|box_start|>…<|box_end|>` span, oolong item lines;
CPT: 512-token pseudo-documents of the prefix):

- **kept set K** = the `gold_fl20p8` rule above (CPT has no gold labels, so it uses `fl20p8`). The
  instruction, query and answer spans are always kept.
- **gold-whole rate q (one global knob)**: with probability q a row keeps its gold documents whole;
  otherwise gold documents get the same `fl20p8` treatment as the rest. This is needed because, on
  id-answer tasks, "the uncompressed documents are the answer" is a learnable shortcut that fails at
  full-context eval. ds64 had to make outlier gold-blind for exactly this reason, which was a per-task
  choice that is now ruled out. A frozen dev-loss probe cannot see this shortcut; only trained runs can.
- **dropped tokens** run blocks [0, L_exit) and are then compacted out (physically removed, original
  RoPE positions kept). Kept tokens run every block.
- **optional slot**: one detached `cent_cmean` slot per compressed document (only meaningful at L_exit = 0).
- **optional layer skip for kept tokens**: stochastic skip of full-attention blocks with prob p (GDN never skipped).

FLOPs ≈ (L_exit / n_layers) · dense + (1 − L_exit / n_layers) · (κ-scaled cost), where κ = |K| / L.
`flop_meter/actual_pflops` is the number of record.

**Full-attention ban lifted (Prasann, 2026-10-16).** Token exit with L_exit > 0 is allowed: the blocks
below the exit run over the full length, and flop_meter must charge that cost. A **TX-local** variant
(document-local attention below the exit) stays available as an optional cheaper arm, not a requirement.

## 2. Shared definitions (both sets)

**Lengths.** 4k, 8k, 16k, 32k, 64k, 128k (CPT: 16k-128k). r256k is left out of the main grid because
its eval needs YaRN x2 + chunked prefill, a confound. One optional point at most.

**CTC classes (fixed composition at every length).** Within setA, only some tasks have eval rungs past 32k
(roster: reorder stops at 16k; grouping, strmatch and xabsence stop at 32k). To keep each class's
composition constant across length:

- **High-CTC** = contradiction (`contradiction_iid`), qdmatch_nq, outlier (scale-k).
  Add grouping, strmatch and xabsence only if their 64k/128k ladders are generated (E9).
- **Low-CTC** = nq, hotpotqa, oolong, rerank.
- reorder is excluded (stops at 16k, and is at chance below 4B).

**α (accuracy target).** Per-task normalized score s = (f1 − chance) / (ref_L − chance), where ref_L is the
canonical dense setA-256k SFT (`q35-4b-dense-ctc-setA-256k…/step1595`) at rung L, re-scored at 500.
The class score is the mean of s over the class's tasks. **Primary α = 0.8** of the fully-trained model's
score at that length; report α = 0.7 and 0.9 as sensitivity checks. Using a normalized α makes the
length axis comparable even though raw f1 falls with length.

**FLOPs.** `flop_meter/actual_pflops` from finished runs only (never `throughput/total petaflops`).
Compare method-vs-method only at identical GPU × micro-batch geometry. Log wall-clock and GPU-hours
next to every point.

**Budgets via WSD branches.** For each (length, class, arm): one warmup-stable run to B_max, with
**decay branches** (10% of each budget's tokens, linear to 0) at B_max × {1/8, 1/4, 1/2, 1}. That gives
4 ladder points for ≈1.2 × B_max tokens. If α is not bracketed, extend the stable run and add a branch;
a crossing is never extrapolated. A cell that cannot reach α is plotted as a lower bound (arrow).

**Eval.** olmo-eval `launch_ctc_suite.py` (branch `prasann/ctc-suite-flopdrop`), `--doc-markers`,
0.8B tokenizer, **full-attention eval for every arm**: the method saves training compute, so the
eval-time model is unchanged. eval_size 500 per task at ≤32k. At 64k/128k use all available rows,
flagged inline. Label every number no-cot. Record every number with `ops result add --check`.

**Seeds.** A second seed at the B/2 point for dense and the chosen method at 8k, 32k and 128k. This
measures run-to-run noise, which the binomial SE does not cover.

## 3. Phase T — tuning on a held-out dev set (shared by both sets)

**Dev sets (never the final eval rows).**

- *SFT dev*: when the per-rung / class-mix shards are repacked (E4), hold out by example id
  **200 examples per task per rung** at 8k and 32k (100 at 64k) from setA_max20_evaliid. These rows are
  IID with training and eval and are not in any training shard. The final olmo-eval suite rows are never
  used for tuning. ⚠ 200/task gives roughly ±0.015 on a class mean: enough to rank configs, not to quote.
- *CPT dev*: dropstruct's `shard_dev` (the stream continued past the training docs). `cpt80` and a fresh
  test shard are reserved for final numbers.

**Dev metrics.** The decision metric is generation f1 on the dev rows. Teacher-forced answer-CE (the
devloss harness) is an early signal only: CE parity ≠ right answer (`memory:ce-vs-chance-floor`).

**Selection score (pre-registered).** For a config at (length, class), take its dev class score and find
the FLOPs that the **dense dev ladder** (3 budgets, log-linear interpolation) needs to reach the same
score. The FLOP multiplier is m = FLOPs_dense(score) / FLOPs_config. **Objective** = geometric mean of m
over {8k, 32k} × {high, low}. **Constraint**: no cell has m < 1 beyond its SE.

| stage | what | size |
|---|---|---|
| T0 | engineering gates: κ = 1 is bit-identical to dense (parity < 1e-6); flop_meter matches the analytic formula; memory drop confirmed by the bench | CPU + 1 GPU |
| T1 | **dev-loss viability check** (devloss harness, frozen setA-32k and setA-256k checkpoints, held-out dev rows, 7 class tasks, 8k / 32k / 64k): `gold_fl20p8_noslot` on all 7 class tasks (fills the missing hotpotqa / qdmatch_nq cells and re-checks rerank at 32k and 64k), and token-exit emulation at L_exit ∈ {0, 8}. Pass = class-pooled median ΔCE ≤ 0.05 and no task worse than +0.3. Failing does not cause a per-task fix; it means changing the global rule | ~5 GPU-h |
| T2-LR | **LR ablation first** (2026-10-19). Every token-drop run so far (ds64, fast2k, fast8k) used LR 5e-6 for dense AND soft arms (setA dense uses 4e-5; the trainer default is 5e-5); it was never swept, and a compressed arm sees fewer real tokens per step, so its best LR may differ from dense's. fast8k screen (8k pool, 2 GPUs, 16M nominal = 123 steps, eval 8k+2k × 500): {dense, `gold_fl20p8_noslot`} × LR {5e-6, 1.5e-5, 5e-5} × {contradiction, nq, outlier, oolong}. Pick ONE LR per arm (shared across tasks), by matched-FLOP f1 pooled over tasks | 24 runs |
| T2a | trained SFT at 8k and 32k, one budget (≈ dense B/4), full setA mix, `gold_fl20p8_noslot` fixed: L_exit ∈ {0, 4, 8, 16} × q ∈ {1.0, 0.5} | 8 configs × 2 lengths |
| T2b | best two from T2a × compress prob {0.5, 1.0} (no slot: it made no difference in the grid; no uniform block skip: it fails rerank). f stays 20% unless every T2a config misses dense at matched FLOPs; then try f ∈ {0.1, 0.35} globally | ≤ 8 configs × 2 lengths |
| T3 | top 3 × 3 seeds at 8k and 32k, plus 1 seed at 64k (length transfer, the learned-router failure mode) | 21 runs |
| CPT-T | first fix the dense-CPT regression (lr 1e-5 vs 3e-5; 4 vs 8 rows/step, keeping the setting where dense CPT beats the base on shard_dev). Then 6 configs at 16k (the SFT winner's L_exit ± one step × f ∈ {0.1, 0.2, 0.35}), gold-blind, scored by shard_dev tail CE at matched FLOPs | ~8 runs |

**Prior (starting point, not a result):** `gold_fl20p8_noslot`, q = 0.5, L_exit = 8. That is
the devloss grid's task-general rule plus dropstruct's best exit. All knobs are global and are selected
on the pooled score; nothing is chosen per task. **Output of Phase T:** one frozen config
**M1** (the FLOP-optimal candidate), plus **M2 and M3**, two more aggressive settings chosen to span
memory for A2. Lower L_exit and lower κ reduce memory; expect them to cost more FLOPs-to-α. All are
recorded in the experiment STATUS before any main-grid launch.

## 4. Set 1 — continued pretraining (CPT)

- **Base**: `q35-4b-base-markerfix`. Re-run the marker cosine + norm audit first.
- **Data**: long-document stream, packed to window L. The prefix is split into 512-token pseudo-documents
  (cpt80 geometry), so "documents" exist without labels. Needs a ≥128k long-document shard (E8). The
  existing `cpt_u1B` shard is 64k rows.
- **Arms**: D = dense CPT (naive, LM loss on all tokens); M1 = frozen TX config, gold-blind (loss on
  kept + tail tokens, as in dropstruct); M2/M3 only at 32k and 128k.
- **Lengths**: 16k, 32k, 64k, 128k (8k optional).
- **Budgets** (B_max, recalibrated by a dense pilot): 16k 256M, 32k 512M, 64k 768M, 128k 1B tokens.
  WSD branches at 1/8, 1/4, 1/2, 1.
- **Metrics**:
  1. *Intrinsic* (every branch): full-context tail CE (last 20%) on the held-out test shard and cpt80.
     α_CE = closing 80% of the gap between the base CE and the best dense CE at that length. This gives a
     CE version of A1 and f1-vs-FLOPs without the class split.
  2. *Downstream, with the class split* (gives the CPT version of A1): from 3 branches per (L, arm) for
     D and M1, run a **fixed dense SFT probe**: the Set 2 dense recipe on the class mix at rung L, at a
     fixed small budget (the Set 2 B/4 point), identical for every CPT checkpoint. Then eval as in §2.
     The x-axis is CPT FLOPs; the probe's cost is constant and is reported but not plotted. An
     end-to-end alternative (method CPT → method SFT, both counted) is decision 9.5.
  3. *Optional cheap proxy*: split tail tokens into long-context-dependent vs local tokens (CE gain from
     extending the context from 4k to L under the base model). Label this a proxy; it is not CTC.
- **Runs**: 4 L × 2 arms + 2 L × 2 arms = 12 stable runs, ~48 branches, ~48 SFT probes.

## 5. Set 2 — setA SFT + CTC eval (the main figures)

- **Base**: `q35-4b-dense-256k-fix/step2385` (the canonical setA base; its marker audit passed).
- **Data**: setA_max20_evaliid, repacked per (rung L, class) with the dev examples held out (E4). Do
  not re-tokenize. Window = L plus template/marker slack (rung rows run slightly over L).
- **Arms**: D = dense SFT (naive), M1 at every length, M2/M3 at 8k, 32k and 128k (for A2).
- **Lengths**: 4k, 8k, 16k, 32k, 64k, 128k. Eval at the matched rung (optionally also at L/2 and 2L).
- **Budgets** (B_max per class mix, recalibrated by a 3-point dense pilot at 8k and 32k from Phase T):
  4k 32M, 8k 64M, 16k 96M, 32k 128M, 64k 192M, 128k 256M tokens. WSD branches at 1/8, 1/4, 1/2, 1.
  ⚠ **The setA pools do NOT hold these budgets past 32k.** setA has an equal token budget per
  (task, rung), about 20M tokens, so example counts halve as the rung doubles: per task 610 at 32k,
  305 at 64k, 152 at 128k, 76 at 256k. A class mix at 128k is about 450-600 examples (60-80M
  tokens), against a B_max of 256M. Long-rung ladders therefore need fresh ctc-data generation (E4
  extension), budgeted before launch.
  Check that rung pools hold B_max with `recipes/eval/setA_audit/count_setA_tokens.py`. If a pool is
  short at 64k/128k, regenerate rows with ctc-data (corpora may be expanded) rather than run multiple
  epochs. A capped pool sets the ladder's ceiling and is flagged.
- **Runs**: 6 L × 2 classes × 2 arms (D, M1) + 3 L × 2 classes × 2 arms (M2, M3) = 36 stable runs,
  144 branches, 144 eval points (plus seeds).
- **Sanity anchor**: one mixed-class D and M1 run at 32k (the canonical full setA mix). This checks that
  class-split training does not change the conclusion.

## 6. Memory benchmark (A2, shared by both sets)

One job per (config, L) for L ∈ {8k, 16k, 32k, 64k, 128k, 256k; 512k/1M with CP}: one forward+backward
step at micro-batch 1. Record `torch.cuda.max_memory_allocated` per rank × CP degree. Two AC settings
(none, full) are reported separately, never mixed. Configs: dense, and TX across a κ × L_exit grid
covering M1-M3. Fit an analytic per-token activation model to these measurements so long lengths that
dense cannot fit can still be plotted (marked as predicted).

Memory falls only for tokens that are **physically removed**. With L_exit > 0, blocks below L_exit hold
full-length activations. That is the memory ↔ FLOPs trade A2 shows. This depends on E1: masking without
compaction saves FLOPs and no memory.

## 7. Deriving each figure

- **f1 vs FLOPs**: per (L, class) panel, plot the class-mean score of each arm's 4 branches against
  actual_pflops. Error bars come from the per-task binomial SE pooled over tasks; seed spread is shown
  where it exists. The grey band in the mock-up is read here as "below the cheapest dense point"
  (decision 9.1).
- **A1**: for each (L, class, arm), the log-FLOPs at which the class score crosses α, by monotone
  (isotonic, then log-linear) interpolation between the bracketing branches. 90% CI from bootstrapping
  eval rows and, where present, seeds. Fit FLOPs ∝ L^β per curve and report β; the claim is β_dense,high
  > β_dense,low ≈ β_ours. Produce one A1 from Set 2 (SFT) and one from Set 1 (CPT probes, plus a CE
  version).
- **A2**: at each length where M1-M3 exist, form the frontier {FLOPs-to-α(config), memory(config)}.
  "FLOP optimal" is the config with minimum FLOPs-to-α. "2× / 4×" is the minimum-memory config whose
  FLOPs-to-α ≤ 2× / 4× that minimum, interpolating κ along the frontier. Memory comes from §6.
  Lengths without M2/M3 ladders use the FLOP ratio measured at the nearest length, marked as such.

## 8. Engineering prerequisites (before Phase T2)

| id | item |
|---|---|
| E1 | Token exit on **packed** setA batches with physical compaction (`cu_seqlens` compact-then-pack). This gives real memory savings and closes ds64's wall-clock gap. `token_exit.py` is on local `prasann/dropstruct` only; port it as variant `tokenexit` in `recipes/train/variants.py` with the parity check |
| E2 | Per-document keep construction for the setA marker format, with gold flags from the gold sidecar (watch the index-base bug) |
| E3 | flop_meter support for token exit (`actual_pflops`), checked against the analytic formula |
| E4 | Per-rung, per-class repacking of setA shards with the held-out dev examples removed, and dev rows exported in the eval's JSONL format |
| E5 | Dev eval path: olmo-eval launcher on the dev rows, plus the devloss answer-CE scorer pointed at them |
| E6 | WSD with decay branches launched from a stable checkpoint in the setA recipe (watch `memory:trainer-silent-autoresume-trap` when branching into a new save folder) |
| E7 | Memory benchmark script (§6) |
| E8 | CPT: fix the dense regression; build a ≥128k long-document shard; build held-out dev and test shards |
| E9 | Optional: generate grouping/strmatch/xabsence ladders (train + eval) at 64k/128k to widen the high-CTC class |

Known trap: `train_ctc_suite.py`'s marker audit dies under torchrun on flopdrop_clean. Run the audit
before torchrun and pass `--skip-marker-audit`.

## 9. Decisions needed from Prasann

1. α definition: normalized to the setA-256k reference with α = 0.8? Also, what the grey band in the f1-vs-FLOPs mock-up marks.
2. Class composition: the fixed 3 high / 4 low tasks, or pay for E9 to add grouping, strmatch and xabsence at 64k+?
3. ~~Gold use in the SFT keep rule~~ — settled 2026-10-16 (Prasann): gold allowed; one `gold_fl20p8`
   approach for all tasks; only global knobs, tuned on pooled dev; the dev-loss probe is used for viability.
4. ~~L_exit > 0 vs the full-attention ban~~ — settled 2026-10-16: the ban is reversed, so plain token exit is allowed.
5. CPT downstream: a fixed dense SFT probe (isolates the CPT effect), or end-to-end method CPT → method SFT?
6. A new experiment page (proposed name `flopdrop_figs`), with this record linked from it.

## 10. Rough cost (H100-hours at ~35% MFU; method arms charged at dense wall-clock until E1 lands)

| block | training | eval / probes |
|---|---|---|
| Phase T (SFT + CPT) | ~80-120 | ~30 |
| Set 2 (setA SFT) | ~150-250 | ~150-300 (144 points) |
| Set 1 (CPT) | ~250-350 | ~100 (48 probes + intrinsic CE) |
| Memory bench | <10 | — |

Total roughly 0.8-1.2k H100-hours. Eval is a large share of it. The main way to reduce it: score the
1/8 branches at 300/task, flagged inline, and score only the bracketing branches at 500.

## 11. Risks

- **Wall-clock**: FLOP wins without GPU-hour wins (ds64). Report both, and treat E1 as required.
- **Train/eval mismatch**: compressed training, full-attention eval. ds64's trained-parity probe saw
  −0.01 to −0.12 own-construction gaps that widen with length. Watch 64k/128k.
- **Gold shortcut**: with gold-forced keep, "the uncompressed document is the answer" is learnable and
  fails at full-context eval. That is why the gold policy is a tuned axis.
- **Length transfer**: configs tuned at 8k/32k may not hold at 128k (learned_router's failure). T3
  includes a 64k check, and the main grid's 128k M1 ladder is itself the test.
- **Small long-rung pools**: eval_size < 500 at 64k/128k is flagged inline. Training pools may cap B_max.
- **Noise**: the binomial SE is ±0.02 per task at 500 rows. A1 crossings inherit it, so report CIs, not
  third decimals.
