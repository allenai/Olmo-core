---
title: Online train-time token router (and mid-stack token exit) refit every step from the training backward
date: 2026-10-22
updated: 2026-10-22
experiment: learned_router
area: token_routing
kind: plan
status: active (overnight screen done: memory/FLOP arms, per-doc floor fixes strmatch; f16 promotion runs in flight)
superseded_by: 
artifact: https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
summary: Online train-time keep router (ridge refit each step on gradient norms) + mid-stack drop; on fresh data the mid-stack router beats dense-s976 on all 4 tasks at 0.61-0.80x its FLOPs (single seed); drop at block 8 (1 and 4 fail), gradient-norm target (activation-strength dh = random), exploration share 0-50% irrelevant.
---
# Online train-time token router

**Goal (Prasann).** A train-time token-drop ALGORITHM, optionally reading a per-document 0/1 gold flag, that is
Pareto-better than dense at matched training FLOPs on all CTC tasks. The best fixed rule so far, `tier67a50m32`
(tokdrop_lr screen), only ties dense at 0.57x FLOPs. Every learned router before this one
(`records/learned-token-router.md`) scored FROZEN checkpoints; this one decides during fine-tuning and learns from
the fine-tuning itself.

**Code.** `src/olmo_core/nn/online_token_router.py` (router), `src/olmo_core/train/callbacks/online_router.py`
(per-step refit + metrics), hooks in `Transformer` (`keep_token_rule="online"`, three small insertions in
`model.py`), FLOP-meter overhead term (`flop_meter.py`), trainer flags `--st-keep-token-rule online --st-otr-*`
(`SoftTokenOptions` fields; `recipes/train/variants_options_test.py` guards them). Tests:
`src/test/nn/online_token_router_test.py`: CPU on horton; GPU on a Beaker H100 (01M5HA8S24PNN2VTZCFFSFMM8C:
79 passed, 0 skipped, including the GDN-hybrid input-prefix and mid-stack modes). Offline probe and launcher:
`research/token_routing/online_router/`.

## Objective (Prasann, 2026-10-22, relayed by the parent session)

"I want the router to be train-time only by the way, and the grad-norm dependence is important as the goal should
be for the truncated forward/backward to be as similar as possible to the un-truncated forward/backward."

* **Train-time only.** The router acts only inside training forwards (`model.training` and soft-token compaction).
  Eval and inference run the full context through every layer.
* **The objective is gradient fidelity.** The truncated pass should reproduce the full pass: answer loss, and above
  all the parameter gradient. The per-token gradient norm is the central target.
* **Measured directly.**
  * Offline: `grad_fidelity.py` gives the cosine, relative error and norm ratio between the truncated and full
    parameter gradients on held-out rows, per rule × budget × drop point.
  * Online: `online_router/fid_mass_*` gives the share of the full-pass squared gradient mass the router's top-k
    keeps.
* **Cost of the full-pass measurement.** It is AMORTISED and CHARGED. Router arms take a full-context step every 8
  steps (`--st-otr-dense-every 8`, the `launch_otr.py` default). These are ordinary dense training steps, so the
  FLOP meter charges them at the full cost. They give unbiased full-pass targets for every token and the
  `fid_mass_*` estimate. On the other steps the targets come from the truncated pass itself: cheap, but measured
  in the compacted context (agreement with the full-pass ranking 0.40–0.66, below).

## Design

**What is routed.** Document body tokens only. Markers, system prompt, query and answer always stay. Dropped
tokens vanish from the row (slot-less compaction, original RoPE positions): the same model path as the
`gold_fl20p8_noslot` / `tier67*` rules, so the comparison with them is like for like.

**Scorer.** Linear, `score_t = w · x_t`, with features

* document structure (58): body offset from start and from end as one-hots 0-15 and log2 buckets, relative
  position, document index, log length, bias;
* lexical (2): the token id occurs in the query; log repeat count in the row;
* hidden state `h` at layer L, RMS-normalised (2,560), and the query interaction `h * q` (2,560) with
  `q` = the mean normalised `h` over the query tokens, plus `cos(h, q)`. With `query_position=after`, document
  tokens never see the query causally, so `h * q` is the only query-aware hidden feature;
* optional gold flag (1), from the shard's gold-fingerprint sidecar (`--st-otr-use-gold`).

Scoring never builds the (T, F) feature matrix: `score = s · w_s + h · (w_h + q * w_hq)`.

**Targets, measured by the training step itself.** For a sample of the kept tokens (all exploration tokens
first), a tensor hook on the compacted embedding output reads `log ||dL/de_t||` (`gnorm`, default) or
`log |<dL/de_t, e_t>|` (`gxh`). Targets are centred per row: only the within-row ranking is fit.

**Fit.** Ridge regression on decayed sufficient statistics (`X^T X`, `X^T y`, decay 0.9 per step), refit before
every optimizer step, after one all-reduce, so every rank holds the same weights. The ridge strength is relative
to the mean Gram diagonal. Default λ = 0.1 (the probe preferred 0.01 at 1M samples; online there are ~30k
samples per step).

**No heuristic init.** Step 1 trains uncompressed (keep-all fallback) and only collects targets over every
token. From step 2 the router routes. Exploration: `explore_frac` (default 10%) of the kept budget is drawn
uniformly at random. These tokens correct the selection bias, since targets exist only for kept tokens, and the
logged held-out rank correlation is measured on them. `--st-otr-dense-every N` adds periodic full-context steps.

**Budget and the gold shortcut.** `keep_frac` of each row's body tokens. The gradient targets are concentrated
on gold documents: on the base model, gold-document AUC of `gnorm` is 0.83-0.94 (below). The parallel
grad-sim analysis (`gradprobe/predict_report_2026-10-21.md`) found that the gradient keep rule has count-leak AUC
≈ 0.95. A router that faithfully imitates gradients would therefore keep gold documents densely and teach the
"dense = gold" shortcut that collapses at full-context eval. Hence `doc_share` (default 1.0): that share of the
budget is spread over documents in proportion to length (each keeps its own top `ceil(doc_share · keep_frac ·
|body|)` tokens by score). The router then decides WHICH tokens of each document stay, not WHICH documents.
Count-leak AUC is 0.5 by construction, up to exploration and rounding. Lower `doc_share` lets the scorer move
budget across documents. `online_router/leak_auc` is logged every routed step whenever the shard has a gold sidecar (it is a diagnostic only; gold is a feature only with `--st-otr-use-gold`).

**Two drop points (the layer-skip half).**

* `--st-otr-drop-layer 0` (input drop): dropped tokens skip every block. Hidden features need a no-grad prefix
  pass through blocks `[0, L)` on the full row (`--st-otr-feature-layer L`). Its cost is
  `(1/3) Σ_{b<L} flops_b(T) · B · T`, about 4% of a dense step at L = 4, and it is charged.
* `--st-otr-drop-layer L` (mid-stack drop): every token runs blocks `[0, L)` on the full row, trained with full
  context. The router then scores the forward's own `h^L` (free, no prefix pass), gathers the kept tokens, and
  the dropped tokens skip blocks `[L, 32)`. This is a two-depth per-(token, layer) skip. Early layers still see
  the whole context: dropped tokens contribute to the kept tokens' representations (and GDN states) below L.
  That keeps the training distribution closer to eval than an input drop at the same downstream budget. The
  cost is `L·Σ_{b<L} f_b(L) + L'·(Σ_{b≥L} f_b(L') + head)` per row, charged as a signed adjustment.
  Composition with the sibling session's whole-block layer drop (`layer_drop.py`) is not wired. Don't enable
  both together.

**FLOP accounting.** The model's compaction counters already give the per-example compacted lengths. Router
overhead goes into `model._flop_extra_actual` and is all-reduced and added to `flop_meter/actual_pflops` by
`FlopMeterCallback`, also logged cumulatively as `flop_meter/overhead_pflops`. That overhead covers the prefix
pass, scoring (2F per body token), the target hook (2D per sampled token), the ridge statistics (2F²N), the
solve (F³/3 + 4F²) and the mid-stack adjustment. The x-axis is therefore honest.

**Diagnostics** (`online_router/*` in wandb and the job log): `fid_mass_router` / `fid_mass_oracle` /
`fid_mass_random` (full-context steps only: share of full-pass squared gradient mass kept by the router's row-level
top-k vs the true top-k vs random), `keep_frac`, `dense`,
`rank_corr_explore` / `rank_corr_kept` (Spearman of the pre-update scores with the measured targets), `leak_auc`,
`fit_samples`, weight norms, `w_in_query`, `w_gold`, `mid_row_frac`.

**Not checkpointed.** The router's statistics and weights live outside the state dict. A resumed run restarts
the router, warmup step included.

## Offline probe (step 2 of the brief): can early-layer features predict the gradient? Yes, partly.

`research/token_routing/online_router/offline_probe.py`, Beaker job 01M5H9AW60673FJG0AXDGC6XMP, commit 5cd8b82.

* **Setup.** Model: the repaired Qwen3.5-4B base (`q35-4b-base-markerfix`, the training start point). Rows: the
  mt4s488 training shard, 48 fit rows per task, then the next 32 per task held out (⚠ eval_size = 32 rows per
  task; per-row metrics averaged). One full-context forward + backward of the answer CE (no-cot, teacher forced)
  per row.
* **Fit.** Ridge on body tokens, pooled over the four tasks. Per-task fits gain only +0.0-0.1 Spearman, so one
  shared router suffices. λ = 0.01 was the best of {0.01, 0.1, 1}.
* **Metrics.** Held-out Spearman / top-10% recall / gradient-mass captured by the predicted top 30%
  (`Σ exp(2·target)`, i.e. squared-gradient-norm share).
* **Baselines.** Random: recall 0.10, mass@30 ≈ 0.30. Oracle mass@30: 0.86-0.95.

Target `gnorm@emb` (what the online router fits), Spearman / recall@10 / mass@30:

| features | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| structure only | +0.30 / 0.30 / 0.71 | +0.25 / 0.38 / 0.82 | +0.26 / 0.44 / 0.74 | +0.16 / 0.19 / 0.49 |
| + lexical | +0.41 / 0.29 / 0.71 | +0.46 / 0.37 / 0.83 | +0.44 / 0.45 / 0.73 | +0.43 / 0.20 / 0.45 |
| + lexical + gold | +0.46 / 0.42 / 0.81 | +0.49 / 0.42 / 0.85 | +0.51 / 0.53 / 0.80 | +0.43 / 0.20 / 0.45 |
| + embedding h⁰ | +0.44 / 0.30 / 0.62 | +0.58 / 0.40 / 0.85 | +0.55 / 0.41 / 0.78 | +0.53 / 0.28 / 0.50 |
| + h⁴, h⁴·q | +0.51 / 0.34 / 0.69 | +0.57 / 0.42 / 0.85 | +0.61 / 0.50 / 0.79 | +0.57 / 0.32 / 0.52 |
| + h⁸, h⁸·q | +0.54 / 0.36 / 0.71 | +0.59 / 0.44 / 0.86 | +0.63 / 0.54 / 0.81 | +0.60 / 0.34 / 0.58 |
| + h⁸, h⁸·q + gold | same as above (±0.01) | | | |

Target `gnorm@L8` (the mid-stack target) with h⁸ + h⁸·q features: Spearman +0.65 / +0.66 / +0.66 / +0.67, mass@30
0.80 / 0.95 / 0.91 / 0.73 (oracle 0.97 / 0.99 / 0.97 / 0.90).

What it says:

1. **Gradient mass is predictable from cheap features.** The top 30% of predicted tokens carry 70-95% of the
   squared gradient norm (random: 30%). Layer-8 features beat layer 4, which beats the embedding, which beats
   structure. The mid-stack drop at L = 8 gets the best features and the most predictable target for free.
2. **The gold flag is redundant once hidden features are in** (≤ 0.01). It helps structure-only scorers (+0.05
   to +0.07). Hidden features recover most of what the flag says, which is consistent with the gradient
   targets' gold AUC (`gnorm@emb` 0.83-0.88, `gnorm@L8` 0.90-0.94): the gradient itself points at gold.
3. **`gxh` is less predictable than `gnorm`** (Spearman 0.3-0.4) but more concentrated (oracle mass@30 0.96-0.99).
   `gnorm` stays the default.
4. **Online targets track full-context ones.** On a random-30% compacted row, `gnorm` of the kept tokens has
   Spearman +0.40 (contradiction), +0.60 (nq), +0.57 (outlier) and +0.66 (oolong) with their full-context
   `gnorm`. That is a usable but noisy signal, and why exploration and per-row centring matter.
5. **Caveat.** This is the BASE model, where CE is unsaturated (CE 0.8-2.7). Mid-training, rows saturate and
   their targets turn to noise (gradprobe found |grad| at g = 1 is noise on near-certain checkpoints). The
   online `rank_corr_explore` metric is the live check.

Recorded with `ops result add --check` (metric prefix `otrprobe_`, run `otr-probe-base-mt4s488`).

## Gradient fidelity of truncated passes (offline, parameter gradient)

`research/token_routing/online_router/grad_fidelity.py`, Beaker 01M5HBH3XHQ45E55RA7XZ3N77X.

* **Setup.** Base markerfix on mt4s488. Held-out rows 80-91 of each task (⚠ eval_size=12 per task), disjoint from
  every fit row. One full pass per row gives `g_full`. Then one truncated pass per rule × budget {0.1, 0.3, 0.5} ×
  drop point {input, mid8}.
* **Metrics.** `cos(g_drop, g_full)` over all parameters and over the blocks only,
  `||g_drop − g_full|| / ||g_full||`, the norm ratio, and ΔCE. The reference is the cosine between two unrelated
  rows' full gradients.
* **Code and data.** Commit 834c93eef; JSON `research/token_routing/online_router/results/gradfid_base/`; ops results
  `gradfid_*`.

Pooled over the four tasks, median over held-out rows (⚠ eval_size = 48 rows, 12 per task). Cells are
cos(g_drop, g_full) / ‖g_drop − g_full‖ / ‖g_full‖ / ΔCE. **Two unrelated rows' full gradients already have
cos 0.40** (nq 0.67, contradiction 0.28), so on the base model raw cosine is inflated by a shared row-independent
direction. Compare against 0.40, not 0. The gold-aware bar rules are not in this table.

| drop point · rule | ρ = 0.1 | ρ = 0.3 | ρ = 0.5 |
|---|---|---|---|
| input · router, per-document share | 0.42 / 1.02 / +0.37 | 0.54 / 0.95 / +0.07 | 0.73 / 0.74 / −0.01 |
| input · router, row top-k | 0.41 / 1.02 / +0.42 | 0.57 / 0.88 / +0.07 | 0.73 / 0.72 / +0.01 |
| input · oracle, per-document | 0.47 / 1.02 / +0.19 | 0.60 / 0.89 / +0.09 | 0.75 / 0.77 / +0.03 |
| input · oracle, row top-k | 0.45 / 0.99 / −0.04 | 0.71 / 0.76 / −0.02 | 0.83 / 0.58 / −0.01 |
| input · random, per-document | 0.39 / 1.04 / +0.47 | 0.46 / 0.97 / +0.40 | 0.51 / 0.92 / +0.32 |
| input · first/last per document | 0.43 / 0.99 / +0.22 | 0.59 / 0.86 / +0.09 | 0.73 / 0.73 / +0.05 |
| **mid8 · router, per-document** | **0.74 / 0.73 / +0.13** | **0.88 / 0.48 / +0.02** | **0.96 / 0.30 / +0.01** |
| mid8 · oracle, per-document | 0.78 / 0.67 / +0.05 | 0.91 / 0.41 / +0.01 | 0.96 / 0.28 / −0.01 |
| mid8 · random, per-document | 0.60 / 0.86 / +0.40 | 0.66 / 0.81 / +0.25 | 0.72 / 0.77 / +0.13 |

Per task at ρ = 0.3, cos for router vs random vs oracle (per-document):

| | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| input | 0.29 / 0.26 / 0.25 | 0.73 / 0.47 / 0.77 | 0.45 / 0.44 / 0.50 | 0.77 / 0.76 / 0.81 |
| mid8 | 0.59 / 0.39 / 0.67 | 0.97 / 0.69 / 0.97 | 0.85 / 0.61 / 0.87 | 0.98 / 0.97 / 0.98 |

What it says:

1. **The mid-stack drop is the big lever for gradient fidelity.** Every token runs blocks 0–7 and dropped tokens
   skip 8–31. At ρ = 0.3 the router keeps cos 0.88 with relative error 0.48, against 0.54 / 0.95 for the same
   router dropping at the input. Mid8 at ρ = 0.1 (0.74 / 0.73) already matches the input drop at ρ = 0.5
   (0.73 / 0.74), and it is cheaper.
   * Approximate forward+backward cost is 8/32 + 24/32 × (kept share of the row).
   * Mid8 at ρ = 0.1 keeps ≈ 15–20% of the row (markers, prompt and answer always stay), so it costs ≈ 0.36–0.40 of
     dense. The input drop at ρ = 0.5 costs ≈ 0.55.
   * The trained runs' `actual_pflops` will give the real numbers.
   * This is consistent with the gradient-similarity record: the gradient arriving at kept tokens is redistributed
     when context is removed early. Keeping the early layers full keeps that signal.
2. **Under mid8 the router is near its ceiling.** Router vs oracle with the same per-document allocation: 0.88 vs
   0.91 at ρ = 0.3, 0.74 vs 0.78 at ρ = 0.1. Random is far behind (0.66, 0.60). The router's value is clearest on
   nq and outlier (+0.28 and +0.24 cos over random at ρ = 0.3) and on contradiction (+0.20).
3. **At the input, the router beats random but not the first/last heuristic** (0.54 vs 0.59 at ρ = 0.3, a tie at
   0.5). Even the oracle gains little over first/last there with per-document allocation (0.60). The large headroom
   at the input is in row-level allocation (oracle row 0.71), which is exactly the gold-concentrating, leaky
   direction. For input drops the per-document constraint costs fidelity. Mid-stack recovers most of it without
   moving budget across documents.
4. **ΔCE tracks fidelity, roughly.** Router and first/last are near-lossless in CE at ρ ≥ 0.3; random is not.
5. **Caveats.**
   * This is the BASE model: the first steps of training, not mid-training (a trained-checkpoint rerun is next).
   * Raw cosine includes the shared direction (unrelated rows 0.40), and centred cosine was not computed.
   * Only 12 rows per task.

### The same probe on a TRAINED checkpoint

* **Run.** Beaker 01M5HQKKR1T184608W6SQ0JY5M, commit 908eb0957, ops results `gradfid_trained_*`.
* **Checkpoint.** `f8k-mt4-otr30L4-s976-tdb-lr2e-06` (244 steps of online-router training on mt4s976).
* **Rows.** Unseen rows of mt4s1464 (fit rows 1000-1031, held-out rows 1080-1091 per task; ⚠ eval_size=12 per task,
  48 pooled). Median full CE is 0.17.
* **The parent's dense checkpoints are unusable.** Its tdb runs crashed in `wandb.finish` before saving any
  checkpoint; their run dirs hold only `flops.json`, `provenance.json`, `run_identity.json` and `wandb/`.

**Unrelated rows now have cos 0.02** (0.40 on the base). The shared direction is gone, so these cosines measure
the row-specific direction directly. Pooled cos / rel err:

| drop point · rule | ρ = 0.1 | ρ = 0.3 | ρ = 0.5 |
|---|---|---|---|
| input · router, per-document | 0.09 / 1.49 | 0.45 / 1.07 | 0.60 / 0.98 |
| input · oracle, per-document | 0.12 / 1.53 | 0.50 / 1.12 | 0.69 / 0.89 |
| input · oracle, row top-k | 0.32 / 1.52 | 0.72 / 0.88 | 0.86 / 0.62 |
| input · random, per-document | 0.07 / 1.44 | 0.06 / 2.46 | 0.06 / 3.48 |
| input · first/last per document | 0.06 / 1.81 | 0.28 / 1.65 | 0.53 / 1.10 |
| **mid8 · router, per-document** | 0.22 / 1.93 | **0.78 / 0.80** | **0.89 / 0.53** |
| mid8 · oracle, per-document | 0.42 / 1.41 | 0.83 / 0.62 | 0.93 / 0.40 |
| mid8 · random, per-document | 0.06 / 2.88 | 0.10 / 2.58 | 0.28 / 1.79 |

Per task at ρ = 0.3 (cos), router / random / oracle, per-document:

| | contradiction | nq | outlier | oolong |
|---|---|---|---|---|
| input | 0.06 / 0.03 / – | 0.40 / 0.03 / – | 0.57 / 0.06 / – | 0.75 / 0.67 / – |
| mid8 | 0.13 / 0.06 / 0.32 | 0.62 / 0.15 / 0.74 | 0.83 / 0.07 / 0.84 | 0.88 / 0.84 / 0.93 |

What changes once the model is trained:

1. **WHICH tokens now matter as much as WHERE.** Random selection loses the gradient direction almost entirely
   (cos 0.06-0.10 at ρ = 0.3, relative error 2.5: the dropped-context gradient is larger than the full one and
   points elsewhere). The router keeps 0.45 at the input and 0.78 mid-stack, close to the oracle (0.83). On the
   base model, random mid-stack still kept 0.66.
2. **The router beats the first/last heuristic once trained** (input, ρ = 0.3: 0.45 vs 0.28; ρ = 0.5: 0.60 vs 0.53).
3. **Mid-stack is still the bigger lever**: 0.78 vs 0.45 at ρ = 0.3. But ρ = 0.1 is too aggressive on a trained
   model (0.22).
4. **Contradiction is the weak task.** Even the mid-stack oracle reaches only 0.32, so what contradiction's
   gradient needs is spread over many tokens.
5. **For the training runs:** early training looks like the base-model table (where random mid-stack is fine), and
   later training looks like this one (where only the router keeps the signal). A random or fixed rule that ties
   the router on short budgets should fall behind on longer ones. That is a prediction for the s1464 runs.

## Smoke training (step 5)

`f8k-mt4-otr30L4-s488-otr-lr2e-06` and companions (Beaker ids below).

* **Recipe.** mt4s488, 122 steps, 16 rows/step, 2 GPUs, LR 2e-6 (the rule arms' LR), input drop.
* **Router.** keep 0.3, doc_share 1, explore 0.1, layer-4 prefix features, ridge 0.1, 2,048 samples per row, a
  full-context step every 8 steps.
* **Runs.** Arms otr30L4 01M5HKH25DRTFX95WZE0E84AFW, rnd30 control 01M5HBRN5SC8E4DVBW848BM5C0, otr30M8
  01M5HBD2K8W5772DDZN4RY1H1J. Earlier launches were cancelled while still pending (stale commit; no full-context steps; then the
  wandb-shutdown crash fixed in 0b5b21ace, which killed the parent's tdb runs after their last step). The launch on
  0b5b21ace died in the trainer's argparse (`--st-keep-token-rule online needs --st-keep-token-k > 0`), fixed in
  c5f8106dc together with `check_variant_wiring.py`.
* **Fresh-data setting** (Prasann, 2026-10-22: at matched FLOPs a drop method gets MORE steps on FRESH rows, same
  16-row batch). Shards mt4s976 (3,904 rows, 244 steps) and mt4s1464 (5,856 rows, 366 steps) are first-N supersets
  of mt4s488. Runs are named to parse in `collect_pareto_points.py` (`-tdb`):
  * input drop: f8k-mt4-otr30L4-s976 / -s1464-tdb-lr2e-06 (01M5HK8GC7NBXWZ3A1RT8DX85P / 01M5HK9TCMGBCF9824ND0CJYF0);
  * mid-stack: otr30M8 s976 / s1464 (01M5HKC241SVX9V8J3FVCK57M1 / 01M5HKD7HD0RHZBG6AC6RB5V9Z);
  * random control: rnd30 s976 / s1464 (01M5HKE537S3Y0Y27XHK1ZKV7M / 01M5HKF5CM1ZFBJ2SA9WC5869M).

  * where-vs-which controls (parent session's request): mid-stack random rnd30M8 (01M5HM848904Y7QG4CSQZH0Y9F /
    01M5HM965QX3VJE61BT3410R9X), mid-stack first/last fl30M8 (01M5HMA8KXE5GZ9V1JJW5X77KY / 01M5HMBKT869Z0H66PEYSMFPHF),
    and the mid-stack router at keep 0.1, otr10M8 (01M5HMCSDHQMCG2S86Z2DMV51F / 01M5HME5F5PKNCFJ6ERSRNSTMA).
    `--st-otr-score random|first_last` swaps the scorer for a fixed one, with the same budget, allocation and drop
    point.

  The s488 rnd30 / otr30M8 runs were cancelled while pending, to put the GPUs here. The comparison is against the
  tdb dense curve (s488 st31 / st61 / 122, s976, s1464) and the tdb rule arms on the same fresh shards.
* **Results** (100 of 104 mt4 scores in, 2026-10-22; only the relaunched rnd30-s1464 contradiction / nq are
  pending).
  * Δ is against dense at matched FLOPs, at 8k.
  * Dense is the best LR per cell. Only LR 5e-6 has ≥ 2 points so far: s488 at 31 / 61 / 122 steps on Lambda and
    fresh-data s976 = 244, s1464 = 366. The `lri` LR-2e-6 curve joins automatically when recorded.
  * Interpolated in log PF on our meter.
  * eval_size 500 per cell, single seed (seed-1 repeats are queued); a difference has an SE of ≈ ±0.02-0.03.
  * Full table: `results/matched_flop_table.md` (`collect_otr.py`). ops results experiment `learned_router`.

| run (fresh data) | PF (× dense-eq) | contradiction | nq | oolong | outlier |
|---|---|---|---|---|---|
| **otr30M8-s1464** mid-stack router | 619 (0.54) | **+0.032** | +0.008 | **+0.038** | **+0.225** |
| **otr10M8-s1464** mid-stack router, keep 0.1 | 471 (0.41) | **+0.043** | +0.016 | +0.021 | **+0.238** |
| rnd30M8-s1464 mid-stack random | 619 (0.54) | +0.016 | +0.018 | +0.021 | +0.148 |
| fl30M8-s1464 mid-stack first/last | 619 (0.54) | +0.022 | −0.028 | +0.035 | +0.137 |
| otr30M8-s976 mid-stack router | 417 (0.54) | +0.029 | +0.010 | −0.022 | +0.049 |
| otr10M8-s976 mid-stack router, keep 0.1 | 319 (0.41) | +0.018 | +0.016 | +0.004 | +0.083 |
| rnd30M8-s976 mid-stack random | 417 (0.54) | −0.019 | +0.016 | −0.009 | +0.053 |
| fl30M8-s976 mid-stack first/last | 417 (0.54) | +0.015 | −0.046 | −0.019 | +0.042 |
| otr30L4-s1464 input router | 516 (0.45) | −0.060 | −0.008 | +0.003 | +0.105 |
| otr30L4-s976 input router | 348 (0.45) | −0.100 | +0.004 | −0.019 | +0.046 |
| rnd30-s1464 input random | 516 (0.45) | pending | pending | +0.002 | +0.021 |
| rnd30-s976 input random | 348 (0.45) | **−0.265** | +0.002 | −0.021 | −0.023 |
| otr30L4-s488 (smoke, same rows as dense s488) | 177 (0.45) | −0.229 | −0.013 | −0.030 | −0.028 |

  **Run-vs-run, no interpolation.** Comparing with actual dense runs avoids assuming a shape for the dense curve,
  which matters on outlier, where dense jumps 0.30 / 0.43 / 0.72 at 122 / 244 / 366 steps. Contradiction / nq /
  oolong / outlier at 8k:

  | run | PF | scores |
  |---|---|---|
  | dense-s976 (Lambda, LR 5e-6) | ≈ 773 | 0.931 / 0.916 / 0.584 / 0.432 |
  | otr30M8-s1464 | 619 (0.80×) | 0.951 / 0.918 / 0.633 / 0.615 |
  | otr10M8-s1464 | 471 (0.61×) | 0.947 / 0.918 / 0.629 / 0.577 |
  | dense-s1464 | ≈ 1,160 | 0.956 / 0.936 / 0.630 / 0.719 |

  * Both mid-stack router runs match or beat dense-s976 on all four tasks with less compute. The outlier gaps
    (+0.18 / +0.15) are ≈ 5-6 SE. The others are 0-1.6 SE, single seed.
  * dense-s1464, at 1.9-2.5× the compute, still leads on nq and outlier.
  * `ops result add --check` flagged otr30M8-s1464 outlier@8k = 0.615 as > 3 SE above the experiment's previous
    best. The eval is clean (500 rows scored, none skipped), and the value sits between dense-s976 and
    dense-s1464, so it is a training effect on a steep curve, not an eval artefact.

  ⚠ **Outlier@2k is confounded by LR.** Dense at LR 5e-6 scores ≈ 0.55 there; dense at LR 2e-6 scores 0.83 at
  122 steps. The +0.2 of the drop arms there is LR, not method, so it is excluded.

  What it says:
  1. **The drop point matters most.** Mid-stack dropping (every token runs blocks 0-7, dropped tokens skip
     8-31) removes the contradiction collapse of input dropping, even with random tokens (−0.265 → −0.019 at
     s976).
  2. **With more training, the selection matters too.** At s976 the mid-stack router and random were close. At
     s1464 the router leads random on contradiction (+0.016), oolong (+0.017) and outlier (+0.077), and leads
     first/last on nq (+0.036) and outlier (+0.088). This is what the trained-checkpoint fidelity probe
     predicted: random keeps cos 0.10 of the gradient there vs the router's 0.78.
  3. **Keep 0.1 mid-stack is as good as keep 0.3 for less compute** (0.41× vs 0.54×). Most of the cost is now
     the 8 full blocks plus the full-context step every 8. A shallower drop layer or fewer dense steps would
     lower it further.
  4. The input router trails dense on contradiction at every budget. Input dropping is not the method.
  5. Single seed throughout. The seed-1 repeats of otr30M8 (s976 / s1464) and the 11-task runs are queued.

### Wall-clock (same cluster: Beaker jupiter, 2 × H100, same base / batch / seq)

`research/token_routing/online_router/wallclock.py`, JSON `results/wallclock.json`. Method: median gap between
consecutive `[step=` log lines (steps ≥ 5), and train time from the first to the last step. Dense is the parent's
tdb Beaker runs; they finished training before their shutdown crash.

| run | s/step | routed / full-context steps | train time | FLOPs × dense-eq |
|---|---|---|---|---|
| dense s488 / s976 / s1464 | 9.10 / 9.13 / 9.20 | – | 18.4 / 37.0 / 56.0 min | 1.00 |
| otr30M8 (mid-stack, keep 0.3) s976 / s1464 | 5.40 / 5.43 | 5.38 / 7.57 | 23.4 / 34.8 min | 0.54 |
| otr10M8 (mid-stack, keep 0.1) s976 / s1464 | 4.89 / 4.84 | 4.85 / 7.58 | 21.7 / 32.0 min | 0.41 |
| rnd30M8 / fl30M8 s1464 | 5.43 / 5.62 | 5.40 / 7.63 | 34.8 / 36.4 min | 0.54 |
| otr30L4 (input drop) s976 / s1464 | 4.71 / 4.74 | 4.67 / 7.92 | 21.1 / 31.4 min | 0.45 |

* **Run-vs-run.** otr30M8-s1464 beats dense-s976 on all four tasks in 34.8 vs 37.0 min (0.94× wall-clock);
  otr10M8-s1464 does it in 32.0 min (0.86×).
* **⚠ The dense arm's wall-clock includes padding.** It trains unpacked rows padded to 10,240 tokens; mean real
  rows are ≈ 7.3k. The soft-token path strips padding and sorts micro-batches by length. A full-context step inside
  a router run therefore takes 7.6 s, not 9.1. The FLOP meter charges only real tokens, so the FLOP comparison is
  fair, but against a pad-free dense step (≈ 7.6 s) the routed steps run at 0.71× (keep 0.3) and 0.64× (keep 0.1)
  of dense time, while their FLOP ratios are 0.54 and 0.41.
* **Where the gap goes.** About a third of the FLOP saving does not become time. Fixed costs per step are FSDP
  collectives, the 8 full-row blocks, the router prefix / features, and the gather. Short compacted rows also use
  GPU kernels less efficiently.
* **What can close it.** The keep-0.1 / block-4 / dense-every-16 arms (queued) cut the full-row share; a CUDA-graph
  / fixed-capacity gather (as in `layerskip/fastskip.py`) would cut the rest.

## Follow-up ablations (2026-10-22, later)

All otr10M8-style (keep 0.1, a full-context step every 8, LR 2e-6, mt4s1464 = 366 steps unless noted). Scores are
8k, eval_size 500, no-cot, recorded with `ops result add --check`. Δ = against the best-LR dense curve at the same PF
(Beaker H100, interpolated: at 471 PF contra 0.904, nq 0.902, oolong 0.608, outlier 0.339). Single seed per cell.
Lambda runs (2 × A100, node02, name `-lro-...-lam`) are compared with a same-cluster otr10M8 run (seed 1).

**1. Split point: a depth threshold at 7 blocks, not a full-attention layer.** Lambda, otr10M*, mt4s1464, 8k,
per-seed scores (s0 / s1 / s2) and the seed mean:

| drop after block | PF | contradiction | nq | oolong | outlier | mean c / n / o / out |
|---|---|---|---|---|---|---|
| 1 | 306 / 310 | 0.767 / 0.763 | 0.882 / 0.870 | 0.617 / 0.616 | 0.322 / 0.314 | 0.765 / 0.876 / 0.617 / 0.318 |
| 4 | 378 / 381 | 0.781 / 0.777 | 0.884 / 0.894 | 0.613 / 0.641 | 0.372 / 0.385 | 0.779 / 0.889 / 0.627 / 0.379 |
| 5 | 400 (s0) | 0.804 | 0.888 | 0.617 | 0.383 | |
| 6 | 422 / 425 | 0.839 / 0.864 | 0.904 / 0.878 | 0.614 / 0.636 | 0.404 / 0.374 | 0.852 / 0.891 / 0.625 / 0.389 |
| **7** | 444 / 447 | 0.929 / 0.925 | 0.902 / 0.912 | 0.621 / 0.665 | 0.558 / 0.604 | **0.927 / 0.907 / 0.643 / 0.581** |
| 8 | 471 (×3) | 0.929 / 0.928 / 0.935 | 0.898 / 0.902 / 0.896 | 0.617 / 0.664 / 0.617 | 0.627 / 0.641 / 0.507 | 0.931 / 0.899 / 0.633 / 0.592 |
| 12 | 565 / 565 | 0.938 / 0.943 | 0.912 / 0.908 | 0.638 / 0.649 | 0.523 / 0.649 | 0.941 / 0.910 / 0.644 / 0.586 |

Beaker at s976 agrees (contradiction M1 0.667, M4 0.728, M8 0.895). Full attention sits at blocks 3, 7, 11, … (3:1
GDN). The hypothesis that block 7's full-attention pass is what matters is FALSIFIED: M7 (blocks 0-6, which excludes
it) equals M8 on every task within seed noise. Contradiction climbs 0.77 → 0.78 → 0.80 → 0.85 → 0.93 from M1 to M7 and
is flat after; outlier stays ≈ 0.32-0.39 through M6 and jumps to ≈ 0.58 at M7. The jump M6 → M7 is the GDN block 6.
M7 is the cheapest split on the plateau (≈ 445 PF vs M8 471 and M12 565). Outlier seed noise here: up to 0.13.

**1b. Frontier: keep 5% and router + layer skip** (Lambda, mt4s1464, seed 0; reference M8 3-seed mean 0.931 / 0.899 /
0.633 / 0.592 at 471 PF):

| run | PF | contradiction | nq | oolong | outlier |
|---|---|---|---|---|---|
| otr05M8 (keep 5%) | 435 | 0.917 | 0.914 | 0.619 | 0.553 |
| otr10M8sk3 (+ every token skips GDN blocks 20 / 24 / 28) | 449 | 0.897 | 0.906 | 0.610 | 0.603 |
| otr10M7 (2 seeds) | 445 | 0.927 | 0.907 | 0.643 | 0.581 |

Keep 5% matches keep 10% within seed noise (contradiction −0.014) but saves only 8% of PF: the fixed cost (blocks
0..L−1 for every token, plus the full-context step every 8) dominates. Router + layer skip (`--train-skip-layers`,
composed in cdab8480a: the skip runs after the mid-stack compaction, and the FLOP adjustment leaves skipped blocks out
of both sides; run_frac 0.906) costs ≈ 0.03 on contradiction for 5% PF. M7 dominates it at the same cost.

**1c. More data at the same keep: otr10M8 on mt4s1952** (1,952 rows / task, 488 steps, Beaker, seed 0): 633 PF
(0.41 × dense-equivalent), contra / nq / oolong / outlier 0.946 / 0.922 / 0.685 / 0.744 at 8k (2k: 0.972 / 0.980 /
0.857 / 0.820). Against dense-s1464 (≈ 1,160 PF: 0.956 / 0.936 / 0.630 / 0.719) it is within 0.01-0.014 on contradiction
/ nq and ahead on oolong (+0.055) and outlier (+0.025) at 55% of the compute. `ops result add --check` flagged outlier
0.744 as above the experiment's previous best: outlier climbs steeply with training (dense 0.30 / 0.43 / 0.72 at 122 /
244 / 366 steps) and this run has 488, so it is a training effect. The matching dense-s1952 point is the parent session's.

**2. Router target (Prasann: "keep important tokens that are already good").** Targets: `gnorm` log ||dL/dh8||;
`gxh` log |<dL/dh8, h8>|; `dh` log ||h8 − h0||, the residual change through blocks 0-7 (activation strength, no backward,
does not vanish when a token is learned); `mix` the within-row mean of the gnorm and dh ranks. Attention mass was not
used (flash attention exposes no weights). Lambda, all at 471 PF:

| target | contra | nq | oolong | outlier |
|---|---|---|---|---|
| gnorm (seed 1) | 0.928 (+0.024) | 0.902 (+0.000) | 0.664 (+0.056) | 0.641 (+0.302) |
| gxh | 0.924 (+0.020) | 0.918 (+0.016) | 0.618 (+0.010) | 0.622 (+0.283) |
| mix | 0.913 (+0.009) | 0.928 (+0.026) | 0.630 (+0.022) | 0.413 (+0.074) |
| dh | 0.890 (−0.014) | 0.928 (+0.026) | 0.622 (+0.014) | 0.359 (+0.020) |

Gradient fidelity of the same choice on the BASE checkpoint (`grad_fidelity.py`, mid8, cos of the parameter gradient
vs the full pass, medians over ⚠ 48 held-out rows only; unrelated rows 0.40):

| keep | gnorm router / oracle | gxh router / oracle | mix router / oracle | dh router / oracle | random |
|---|---|---|---|---|---|
| 0.1 | 0.739 / 0.775 | 0.751 / 0.754 | 0.669 / 0.587 | 0.584 / 0.574 | 0.597 |
| 0.3 | 0.879 / 0.914 | 0.907 / 0.870 | 0.791 / 0.662 | 0.605 / 0.609 | 0.661 |

The same on the TRAINED otr30L4-s976 export (unseen mt4s1464 rows, ⚠ 48 rows; unrelated rows 0.02): keep 0.3 gnorm
0.776 / 0.823, gxh 0.799 / 0.740, mix 0.659 / 0.271, dh 0.176 / 0.143, random 0.099; keep 0.1 gnorm 0.224 / 0.423, gxh
0.182 / 0.286, mix 0.394 / 0.100, dh 0.098 / 0.126, random 0.057. At 0.3 the ordering holds; at 0.1 every rule loses
most of the direction on this (input-drop-trained) checkpoint and the mix router reads highest, mostly outlier (0.53
vs 0.22, 12 rows) -- against the mix training run's outlier 0.413, so treated as noise until a mid-stack-trained probe
repeats it. JSON `results/gradfid_tgtabl_{base,otr30L4s976}/`, ops `otr-gradfid-tgtabl-*`.

Spearman of the targets with gnorm over body tokens (base): gxh +0.64, mix +0.64, dh −0.165 (trained: +0.66 / +0.66 / −0.10). Fidelity and training agree:
gnorm ≈ gxh > mix > dh ≈ random. The tokens with the largest early residual change are not the ones the gradient
needs (dh is slightly anti-correlated with it), and on outlier, where the needed token is the one that does not fit,
dh throws away nearly all of the router's gain. A fit-free "top-k by dh" rule was queued (dh10M8 / dh10M8de0,
`--st-otr-score dh`) and cancelled at 33 min because its fidelity is below random. Keep gnorm.

**3. Exploration share** (otr10M8, s1464, 471 PF): 0% 0.919 / 0.898 / 0.621 / 0.595; 10% 0.947 / 0.918 / 0.629 /
0.577; 30% 0.929 / 0.906 / 0.623 / 0.644; 50% 0.934 / 0.898 / 0.630 / 0.592; 100% (random) 0.922 / 0.910 / 0.635 /
0.423. 0-50% are within noise; selection bias from exploiting the router does no harm. Random loses ≈ 0.17 on outlier.

**4. Fewer full-context steps (de16) does not pay.** otr10M8de16-s976 (287 PF) 0.877 / 0.884 / 0.601 / 0.340 is
roughly dense-matched, vs otr10M8-s976 (319 PF) +0.018 / +0.016 / +0.004 / +0.083. otr10M4de16 is below dense on
contradiction at both budgets.

**5. Seed noise.** otr30M8sd1-s976 0.917 / 0.916 / 0.619 / 0.494 vs seed 0 0.926 / 0.908 / 0.591 / 0.365: outlier
moves 0.13 between seeds at s976, contradiction / nq stay within 0.01. Single-seed outlier differences below ≈ 0.15
at s976 do not count. (s1464 repeat evaluating.)

## Aggressive cuts (2026-10-23, overnight; Prasann: "much more aggressive cuts to computation")

otr10M7 costs ≈ 0.39 × dense, and the FLOORS dominate: blocks 0-6 on every token (≈ 22% of dense), the full-context
step every 8 (≈ 12%), router overhead (a few %), the routed blocks 7-31 at 10% (≈ 8%). New opt-in levers, each charged
exactly what runs and covered by a CPU test (`src/test/nn/online_token_router_test.py`):

| flag | what it does | FLOP charge |
|---|---|---|
| `--st-otr-early-nograd` | routed steps run blocks [0, L) under no_grad (a fresh leaf at the split keeps the gnorm hook); those blocks learn only on dense steps | blocks [0, L) forward-only (1/3) on routed steps |
| `--st-otr-dense-tail-frac F` | the last F of the steps run uncompressed; with `--st-otr-dense-every 0` = routed, then a dense tail | dense steps charged in full |
| `--st-otr-stages "3:0.5,7:0.1,16:0.03"` | pyramid: a chain of routers, each with its own ridge fit and targets at its own layer; keeps are fractions of the original body | stage s: Lk·Σ(blocks ≥ L_s at Lk) − L_in·Σ(blocks ≥ L_s at L_in) |
| `--st-otr-merge-dropped` | each document's dropped tokens averaged into one unlabelled summary token after its last dropped token (causal) | summaries count as kept tokens, plus 2·D per dropped token |
| `--st-otr-keep-first K` | each document's first K body tokens always kept (header ablation) | none |

**Gradient-composition prior** (the grad-composition session, ⚠ 64 held-out mt4s1952 rows / task, full-context answer
CE): blocks 0-6 carry 0.42-0.51 of the per-row parameter-gradient energy at base and 0.13-0.25 after 366 dense steps
(0.29-0.34 of the minibatch-summed gradient), so freezing / no-grad early blocks gives up a real share of learning,
largest at the start. Per-token importance is stable early and drifts late (Spearman of ||dL/dh||, base -> trained:
L3~L7 0.88 -> 0.88, L7~L16 0.78 -> 0.60, L7~L24 0.57 -> 0.35; the last block is anti-correlated with L7, -0.19).
For the pyramid: 0.93-0.96 of the later layer's G^2 sits inside the top 50% at L3 (the 3 -> 7 stage is nearly free),
and the nested 50%@3 -> 10%@7 -> 3%@16 captures 0.69-0.81 of G24^2 vs 0.61-0.75 for top-3% at L7 alone (rescoring at
16 is worth ≈ +0.08 of mass); random 0.03.

**Revision (Prasann, 2026-10-23: "ideas which can save both on FLOPs and memory"; "validate at smaller scale first").**
The dense tail is dropped and every arm runs WITHOUT full-context steps (`--st-otr-dense-every 0`); new
`--st-otr-freeze-early` (embeddings + blocks [0, L) frozen: no grads, no optimizer state, forward-only every step);
`online_router/peak_mem_gib` logged per step. Screen at mt4s488 (122 steps) first, promote only arms that tie or beat
otr10M7-s488: Beaker `f8k-mt4-{otr10M7,otr10M7de0,ng0M7,frz0M7,pyr3de0,pyr3frz0,mrg05M7de0}-s488-tdb-lr2e-06`, Lambda
`f8k-mt4s488-{frz0M7,ng0M7,pyr3frz0,mrg05M7de0}-lro-lr2e-06-lam`. The s1464 tail / de8-pyramid / de8-merge runs below were
cancelled before starting.

Runs (mt4s1464, LR 2e-6, seed 0): Lambda `f8k-mt4s1464-otr10M7{ng,tail,de0,ngtail}-lro-lr2e-06-lam`; Beaker
`f8k-mt4-{pyr3,pyr3ng,pyr2,mrg05M7,mrg10M7}-s1464-tdb-lr2e-06`; plus the allocation / header ablation
`f8k-mt4-otr10M7{,ds0,ds50,hdr8}-s1464-tdb` and the same on mt11s720. Results pending.

### Screen results (2026-10-23; single seed; eval_size 500; all recorded with `ops result add --check`)

**mt4s488 (122 steps), Lambda, 8k contra / nq / oolong / outlier.** Dense s488 at 31 steps (≈ 99 PF): 0.525-0.669 /
0.82 / 0.50 / 0.23-0.28.

| arm | PF (× 389 dense-eq) | peak GiB / GPU | contra | nq | oolong | outlier |
|---|---|---|---|---|---|---|
| otr10M7 (de8, reference) | 152 (0.39) | 60.1 | 0.705 | 0.860 | 0.564 | 0.335 |
| otr10M7de0 (no full steps) | 120 (0.31) | 60.1 | 0.663 | 0.846 | 0.542 | 0.343 |
| ng0M7 (early blocks no-grad) | 73 (0.19) | 53.0 | 0.426 | 0.764 | 0.542 | 0.275 |
| frz0M7 (early stack frozen) | 72 (0.19) | 47.7 | 0.415 | 0.744 | 0.537 | 0.271 |
| **fa30M7** (freeze after 30 steps) | 84 (0.22) | 47.7 after step 31 | 0.589 | 0.852 | 0.556 | 0.264 |
| fa61M7 (freeze after 61 steps) | 96 (0.25) | 47.7 after step 62 | 0.616 | 0.862 | 0.513 | 0.342 |
| pyr3frz0 (pyramid + frozen 0-2) | 66 (0.17) | 53.4 | 0.336 | 0.842 | 0.511 | 0.265 |
| mrg05M7de0 (merge, keep 5%) | 109 (0.28) | 60 | 0.551 | 0.810 | 0.517 | 0.314 |
| otr10M7f8 (per-doc floor 8) | 160 (0.41) | | 0.783 | 0.858 | 0.580 | 0.348 |
| otr15M7 (keep 0.15, control) | 165 (0.42) | | 0.744 | 0.864 | 0.580 | 0.377 |
| otr10M7f16 (floor 16) | 180 (0.46) | | 0.815 | 0.866 | 0.605 | 0.333 |
| otr20M7 (keep 0.20, control) | 177 (0.46) | | 0.780 | 0.888 | 0.575 | 0.292 |

Beaker s488 copies agree (otr10M7 0.702 / 0.850 / 0.586 / 0.302; de0 0.679 / 0.848 / 0.561 / 0.317; pyr3de0 0.418 /
0.858 / 0.506 / 0.333). Findings: (1) the early stack must keep learning -- cutting its gradient (no-grad or frozen from
step 0) costs ≈ 0.24 contra and 0.08-0.1 nq for −40% PF and −7 to −12 GiB; consistent with the grad-composition prior
(blocks 0-6 hold 0.42-0.51 of the gradient energy at base). (2) warmup-then-freeze recovers most of it: fa30M7 keeps
nq at de0's level and contra at 0.59 for −30% PF and −12 GiB vs de0. (3) removing the full-context steps (de0) costs
≈ 0.04 contra for −21% PF. (4) the pyramid loses contradiction (keeping 50% at block 3 starves it). (5) merge at 5% is
not better than drop at 10%. (6) the per-document floor gains ≈ +0.035-0.04 contra over a keep-matched control.

**mt11s720 (Beaker), 8k, mean over the 9 tasks (reorder@8k and rerank excluded):**

| arm | PF (× dense) | strmatch | grouping | xabsence | contra | mean9 |
|---|---|---|---|---|---|---|
| otr10M7 | 295 (0.41) | 0.789 | 0.287 | 0.523 | 0.835 | 0.699 |
| otr10M7f8 | 313 (0.44) | 0.800 | 0.277 | 0.561 | 0.833 | 0.699 |
| **otr10M7f16** (seed 0) | 352 (0.49) | **0.922** | 0.270 | 0.595 | 0.864 | **0.724** |
| **otr10M7f16** (seed 1) | 353 (0.49) | **0.907** | 0.277 | 0.631 | 0.874 | **0.735** |
| **fa247f16M7** (freeze at half + floor 16, no full steps) | 258 (0.36) | **0.905** | 0.260 | 0.584 | 0.866 | 0.712 |
| fa247f16M7 seed 1 | 258 (0.36) | 0.920 | 0.284 | 0.632 | 0.851 | 0.722 |
| **fa247f16M7, LR 5e-6** | 258 (0.36) | **0.933** | 0.283 | 0.686 | 0.897 | **0.750** |
| prog4712f16M7 (progressive freeze + floor 16), LR 5e-6 | 243 (0.34) | 0.928 | 0.273 | 0.671 | 0.893 | 0.731 |
| prog4712f16M7pt (+ per-task routers), LR 5e-6 | 244 (0.34) | 0.951 | 0.309 | 0.637 | 0.901 | 0.737 |
| lora16f16M7 (LoRA r16 early stack + floor 16), LR 5e-6 | 260 (0.36) | 0.827 | 0.254 | 0.588 | 0.907 | 0.716 |
| **lora64f16M7 (LoRA r64 early stack + floor 16), LR 5e-6** | 265 (0.37) | **0.927** | 0.284 | **0.738** | 0.909 | **0.748** |
| lora64fa371f16M7 (LoRA r64 + floor 16, frozen at 75%), LR 5e-6 | 253 (0.35) | 0.952 | 0.274 | 0.617 | 0.893 | 0.735 |
| otr10M7f16sk3 (+ skip layers 20,24,28), LR 5e-6 | 286 (0.40) | 0.894 | 0.241 | 0.621 | 0.887 | 0.735 |
| r20to7f16 (stages 7:0.2,16:0.07 + floor 16), LR 5e-6 | 310 (0.43) | 0.943 | 0.292 | 0.618 | 0.922 | 0.748 |
| lora64fa371r20to7f16 (LoRA r64 + freeze 75% + two-stage + floor 16), LR 5e-6 | 260 (0.36) | 0.935 | 0.307 | 0.544 | 0.907 | 0.738 |
| lora64fa371r20to10f16 (same, stage 2 at 10%), LR 5e-6 | 262 (0.36) | 0.907 | 0.322 | 0.607 | 0.901 | 0.738 |
| lora64fa420r20to7f16 (LoRA r64 + two-stage + floor, freeze at 85%), LR 5e-6 | 265 (0.37) | 0.927 | 0.305 | 0.584 | 0.920 | 0.740 |
| **lora64r20to7f16 (LoRA r64 + two-stage + floor, NO freeze), LR 5e-6**, seeds 0 / 1 | 273 (0.38) | **0.947 / 0.942** | 0.285 / 0.295 | **0.683 / 0.708** | 0.909 / 0.892 | **0.752 / 0.750** |
| lora64r20to10f16 (same, stage 2 at 10%), LR 5e-6 | 274 (0.38) | 0.929 | 0.276 | 0.645 | 0.909 | 0.740 |
| otr10M7hdr8 (first 8 tokens kept) | 314 (0.44) | 0.744 | 0.306 | 0.648 | 0.835 | 0.711 |
| otr10M7ds0 / ds50 (global budget) | 293 | 0.770 / 0.769 | | | | 0.696 / 0.697 |
| rnd10M7 (random) | 295 | 0.793 | 0.353 | 0.466 | 0.803 | 0.691 |
| otr30M8 | 400 (0.56) | 0.845 | 0.301 | 0.628 | 0.876 | 0.728 |
| dense-s720 | 1.0 | 0.974 | 0.361 | 0.555 | 0.873 | 0.741 |

Strmatch: random = router (0.793 / 0.789) -- the loss is the cut, not the choice, as the fidelity probe showed (block 7,
keep 0.3, trained otr10M8-s480: router 0.08 / oracle 0.55 / random 0.09 on strmatch vs 0.89 / 0.90 / 0.15 on grouping;
⚠ 12 rows / task). A per-document floor of 16 tokens fixes it (0.922; `--check` flags the new high) and brings the mean
to otr30M8's at 12% less compute; seed 1 replicates it (strmatch 0.907, mean 0.735; 2-seed mean 0.730 vs dense 0.741 at 0.49x). Combining both levers (fa247f16M7: early stack frozen after 247 of 495 steps, optimizer state released, 63.3 -> 50.9 GiB on H100; floor 16; no full steps) keeps the fix (strmatch 0.905) at 0.36x: mean 0.712, above otr10M7's 0.699 at 0.41x. Seed 1 replicates (strmatch 0.920, mean 0.722; 2-seed mean 0.717). At LR 5e-6 the same arm scores mean 0.750
(strmatch 0.933, xabsence 0.686, contra 0.897) at 0.36x -- above dense-s720 at 2e-6 (0.741), but dense mt11 has only
been run at 2e-6 in this table. **LR-matched dense exists (parent, Lambda `f8k-mt11s{240,480,720}-dense-lrg-lr5e-06-lam`,
research/token_routing/ds64_fast8k/results/pareto_points.json): mean9 0.697 @ 232 PF, 0.750 @ 483 PF, 0.788 @ 715 PF.
So fa247f16M7 at 5e-6 (0.750 at 258 PF) is +0.047 over dense interpolated to its PF and matches dense-s480 at 0.53x
its FLOPs** (−0.038 vs dense on the same 720 rows/task; single seed, eval_size 500 per task). **New levers on mt11s720 at 5e-6 (seed 0, vs dense interpolated to their PF ≈ 0.699-0.703):** progressive
freeze + floor 0.731 (+0.032), + per-task routers 0.737 (+0.038; strmatch +0.023 and grouping +0.036 over the shared
router -- the tasks per-task routing was meant to help, single seed), LoRA r16 + floor 0.716 (+0.013; strmatch 0.827).
None beats the single freeze fa247f16M7 (0.750, +0.047). **LoRA rank 64 + floor (lora64f16M7) ties it: mean9 0.748 at
0.37x (+0.044 vs dense interpolated to 265 PF), strmatch 0.927, xabsence 0.738 -- the same mt11 quality as the best freeze
arm with the ~48 GiB peak of the LoRA stack (vs 60 GiB for any warmup-then-freeze arm before its freeze).** Rank 16's
strmatch collapse (0.827) was the rank, not LoRA. Freezing that LoRA stack at 75% (lora64fa371f16M7) gives mean9 0.735 at 0.35x (+0.034 vs dense
interpolated to 253 PF): -0.013 vs unfrozen LoRA, almost all of it xabsence (0.617 vs 0.738; strmatch up to 0.952) --
single seed, one task, so not a call that freezing hurts the 11-task mix. The parent's s1952 winners on mt11s720 (floor 16, seed 0): sk3 0.735 at 0.40x (+0.027 vs dense
interpolated to 286 PF), r20to7 0.748 at 0.43x (+0.035 at 310 PF) -- both above dense at matched FLOPs, neither above
fa247f16M7 (0.750 at 0.36x) or lora64f16M7 (0.748 at 0.37x), which reach the same mean9 for less compute. The mt4 best combination (LoRA r64 + freeze 75% +
two-stage) on mt11s720 + floor: 0.738 at 0.36x (+0.035 vs dense interpolated to 260 PF) -- below fa247f16M7 / lora64f16M7.
Both LoRA-then-freeze runs score low on xabsence (0.544 / 0.617 vs 0.738 unfrozen LoRA), so on this mix the FREEZE, not
the two-stage drop, looks like the cost (single seed each). Stage 2 at 10% on mt11 gives the same 0.738 (xabsence 0.607):
three LoRA-then-freeze runs on mt11 all sit at xabsence 0.54-0.62 vs 0.738 unfrozen. No-freeze / freeze-85% runs decide it. On mt4s1464 the freeze is free: LoRA r64 + two-stage without it
(lora64r20to7) = 0.944 / 0.916 / 0.615 / 0.713 at 0.26x vs 0.938 / 0.922 / 0.610 / 0.765 frozen at 75% (0.24x). **The mt11
freeze test settles it: freezing the LoRA stack costs xabsence on the 11-task mix whether early or late (freeze 75%:
0.544, freeze 85%: 0.584, mean9 0.738 / 0.740), and without the freeze xabsence comes back (0.683) and LoRA r64 +
two-stage + floor is the best mt11 point so far: mean9 0.752 at 0.38x, +0.046 vs dense interpolated to 273 PF. Universal
recipe: LoRA r64 early stack + two-stage drop (7:0.2,16:0.07), NO freeze -- 0.26x on mt4 (= the frozen variant within
noise), 0.38x on mt11 (+ floor 16); ~0.02x more than freezing, which is free only on mt4.** Recipe on mt4s1464, 2 seeds: 0.945 / 0.912 / 0.636 /
0.726 at 0.26x, above the one-stage router (0.937 / 0.902 / 0.623 / 0.682 at 0.30x, 60 GiB) on every task at ~49 GiB.
(Seed 1's contradiction@2k came from the eval log, 0.96 at 3 s.f.: the Lambda watcher recorded once all four multirung
jsons existed, but a json can land after its first rung, and the checkpoint-dir delete then removed the node-local 2k
output. The watcher now waits for both rungs in every json.) **mt11 recipe, seed 1: mean9 0.750 (xabsence 0.708) -> 2-seed 0.751 at
0.38x, +0.045 vs dense interpolated to 273 PF: level with the best mt11 arm (fa247f16M7, 0.750 at 0.36x, 60 GiB peak) at
the LoRA memory peak. Stage 2 at 10% does not carry to mt11 (0.740).** Global budget: no collapse (leak_auc 0.58 vs 0.49 per-document) but no gain.
Promoted: otr10M7f8 / f16 to mt4s1464 (Lambda), f16 seed 1 on mt11s720. otr10M7f8 at mt4s1464 (467 PF): 0.931 / 0.892 /
0.616 / 0.679 vs otr10M7 (2 seeds, ≈ 445 PF) 0.927 / 0.907 / 0.643 / 0.581 -- with more training the floor's mt4 gain
vanishes (contra level; the rest within seed noise): it matters for tiny documents (strmatch) and early in training.
otr10M7f16 at mt4s1464 (527 PF): 0.950 / 0.894 / 0.636 / 0.580 -- +0.023 contra for +18% PF, on the compute curve
(otr30M8, 619 PF: 0.951 / 0.918 / 0.633 / 0.615), not above it.
**Warmup-then-freeze at mt4s1464:** fa183M7 (no full steps, early stack frozen after 183 of 366 steps; 275 PF = 0.24x
dense-eq; 47.7 GiB after the freeze) 0.898 / 0.884 / 0.647 / 0.556, vs otr10M7 (2 seeds, 445 PF = 0.39x) 0.927 / 0.907 /
0.643 / 0.581: -38% PF and -12 GiB for the second half at -0.03 contra. Against the best-LR dense curve interpolated
at 275 PF (≈ 0.885 / 0.897 / 0.569 / 0.28, between dense-s488 61 and 122 steps) it is level on contra / nq and ahead on
oolong / outlier.
**Freeze-point sweep + LR at mt4s1464 (Lambda, de0, no full steps), 8k contra / nq / oolong / outlier:**

| run | PF (x 1154 dense-eq) | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| fa92M7 (freeze at 25%) | 240 (0.21) | 0.878 | 0.882 | 0.618 | 0.488 |
| fa183M7 (50%), seeds 0 / 1 | 275 (0.24) | 0.898 / 0.881 | 0.884 / 0.890 | 0.647 / 0.666 | 0.556 / 0.491 |
| fa275M7 (75%) | 311 (0.27) | 0.919 | 0.888 | 0.637 | 0.553 |
| **fa183M7, LR 5e-6**, seeds 0 / 1 | 275 (0.24) | **0.920 / 0.927** | 0.884 / 0.916 | 0.624 / 0.593 | **0.622 / 0.691** |
| fa92M7, LR 5e-6 | 240 (0.21) | 0.913 | 0.908 | 0.618 | 0.744 |
| fa275M7, LR 5e-6, seeds 0 / 1 | 311 (0.27) | 0.936 / 0.923 | 0.896 / 0.888 | 0.626 / 0.630 | 0.664 / 0.627 |
| otr05M7 de0 (keep 5%, no freeze), LR 5e-6 | 304 (0.26) | 0.906 | 0.890 | 0.589 | 0.460 |
| otr10M7 de8, LR 2e-6 (2 seeds) | 445 (0.39) | 0.927 | 0.907 | 0.643 | 0.581 |
| **otr10M7 de0 (no freeze), LR 5e-6**, seeds 0 / 1 | 346 (0.30) | **0.947 / 0.926** | 0.902 / 0.902 | 0.610 / 0.636 | 0.673 / 0.690 |
| **otr10M7 de8, LR 5e-6**, seeds 0 / 1 | 445 (0.38) | **0.957 / 0.953** | 0.912 / 0.914 | 0.616 / 0.627 | 0.703 / 0.749 |
| otr10M7 de16 (full step every 16), LR 5e-6, seeds 0 / 1 | 395 (0.34) | 0.939 / 0.952 | 0.918 / 0.906 | 0.607 / 0.643 | 0.709 / 0.741 |
| fa275M7 de8 (freeze 75% + full step every 8), LR 5e-6, seeds 0 / 1 | 410 (0.35) | 0.957 / 0.933 | 0.918 / 0.908 | 0.625 / 0.643 | 0.652 / 0.684 |
| fa183M7 de8 (freeze 50% + full step every 8), LR 5e-6, seeds 0 / 1 | 375 (0.32) | 0.941 / 0.950 | 0.904 / 0.916 | 0.618 / 0.633 | 0.513 / 0.699 |
| dense-s1464, LR 5e-6 | ≈ 1154 (1.0) | 0.956 | 0.936 | 0.630 | 0.719 |

A later freeze buys contradiction smoothly; LR 5e-6 (the 64k study's router LR) matches otr10M7 at 0.24x on two seeds:
mean 0.924 / 0.900 / 0.609 / 0.657 vs 0.927 / 0.907 / 0.643 / 0.581 at 0.39x (outlier +0.08, oolong -0.03; outlier@2k
0.68-0.69 is the known LR confound on that cell). At LR 5e-6 every freeze point (25 / 50 / 75%, 0.21-0.27x) is at or above otr10M7-at-2e-6 on contra / nq. **LR-matched references (5e-6, 2 seeds each) settle it: most of that gain is the LR, not the freeze.** otr10M7 de8 at
5e-6 (0.38x), 2-seed mean 0.955 / 0.913 / 0.622 / 0.726 vs dense-s1464 0.956 / 0.936 / 0.630 / 0.719: **level with dense
on contra / oolong / outlier at 0.38x, -0.023 on nq** -- the router's best point yet. Dropping full steps (de0, 0.30x)
mean 0.937 / 0.902 / 0.623 / 0.682 (-0.018 contra, -0.044 outlier); freezing on top costs -0.007 / -0.010 / +0.005 /
-0.036 at 75% (fa275 2-seed mean 0.930 / 0.892 / 0.628 / 0.646, 0.27x) and -0.013 contra at 50% (fa183 2-seed mean
0.924, 0.24x). **At matched FLOPs, freezing beats a smaller budget:** keep 5% instead of 10% (otr05M7de0, 304 PF, 0.26x)
scores 0.906 / 0.890 / 0.589 / 0.460, below both freeze arms on every task (outlier -0.19) -- with blocks 0-6 running on
every token, halving the keep fraction saves only 42 PF, while freezing removes the early stack's backward.
**Spacing the full steps and combining them with the freeze (LR 5e-6, 2-seed means):**

| arm | x dense | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| otr10M7 de8 | 0.38 | 0.955 | 0.913 | 0.622 | 0.726 |
| fa275M7 de8 | 0.35 | 0.945 | 0.913 | 0.634 | 0.668 |
| **otr10M7 de16** | **0.34** | **0.946** | **0.912** | **0.625** | **0.725** |
| fa183M7 de8 | 0.32 | 0.946 | 0.910 | 0.626 | 0.606 |
| otr10M7 de0 | 0.30 | 0.937 | 0.902 | 0.623 | 0.682 |
| fa275M7 de0 | 0.27 | 0.930 | 0.892 | 0.628 | 0.646 |
| fa183M7 de0 | 0.24 | 0.924 | 0.900 | 0.609 | 0.657 |
| dense-s1464 | 1.0 | 0.956 | 0.936 | 0.630 | 0.719 |

A full step every 16 keeps every task of every-8 except -0.009 contra at 0.34x: the better point. Freeze + every-8
lands on the same contra / nq / oolong at 0.32-0.35x; outlier there is too noisy to call (fa183de8 seeds 0.513 / 0.699,
a 0.19 spread -- outlier@8k is the noisiest cell, so read single outlier numbers with care). PF of the freeze + full-step
arms is exact: after the freeze every training forward (full steps included) runs blocks [0, L) forward-only and is
charged 1/3 there. Contra / nq trade smoothly with
PF along the whole ladder; at ≤ 0.30x it costs 0.02-0.03 contra. Running: de32 and fa275de16 (seed 0). At
matched LR the ladder is one smooth curve: the full-context steps buy contra / outlier, freeze is a cheap
compute+memory knob (-12 GiB after the freeze). Seed spread on de0 contra is 0.021, larger than the ±0.01 eval SE.
The freeze point trades smoothly: fa30 / fa61 / no freeze = contra 0.589 / 0.616 / 0.663 at 84 / 96 / 120 PF.

## Long-context screen (ds64 setting) -- CANCELLED 2026-10-22

Cancelled before any started (Prasann: reproduce the per-task numbers instead, since the mixed runs give fewer
per-task datapoints). Per-task runs on the fast8k per-task pools (`ds64/fast8k/shards/<task>_g16M`, oolong `_g16M_eg`,
122 steps, query after): `f8k-<task>-otr10M8-16M-tdp-lr2e-06` for all four tasks plus otr30M8 on contradiction and
outlier, evals running. The original ds64 plan, for reference:

`research/token_routing/online_router/launch_ds64_otr.py`, state `state_ds64.json`. The parent session's reason:
* At 8k, otr10M8-s1464 recovers about half of ds64's per-task wins. ds64 wins grow with length (contradiction +0.11
  / +0.18 / +0.19 at 16k / 32k / 64k), and the mid-stack cost falls slowly with length, so the long-context
  comparison is the one that matters.

Setup:
* **Geometry.** ds64's soft-arm geometry: per-task short-heavy 2k-56k pools, UNPACKED rows at seq 65,536, 128
  rows/step, flash_2, LR 5e-6, repaired 4B base. ds64 only packs the dense arm; its soft arms train unpacked, so
  segment mode is not needed.
* **Micro-batch 1.** Mid-stack runs blocks 0-7 on the full row and every 8th step is fully dense.
* **8 GPUs, as in ds64.** A first launch at 4 GPUs was cancelled while pending: 4-GPU requests schedule slowest on jupiter.
* **512 target samples per row.**
* **Evals.** The ds64 orchestrator's 5-rung files (2k / 8k / 16k / 32k / 64k, 500 rows, query after).
* **Reference.** The ds64 dense curves in ops results (experiment `ds64`): contradiction 16M / 32M / 64M, nq 16M /
  32M / 64M / 128M.

Runs:

| run | Beaker |
|---|---|
| ds64-contradiction-otr10M8-b128f-u64M | 01M5JEFZ3YBD22MBQ9SPXWFTVS |
| ds64-contradiction-otr30M8-b128f-u64M | 01M5JEGS0S6MG26DH6J32YTCZJ |
| ds64-nq-otr10M8-b128f-u64M | 01M5JEHJGK8Q230Q5HHPXY3N4F |
| ds64-nq-otr30M8-b128f-u64M | 01M5JEJDDJD79EF03EASST2QCG |

## Proposed run matrix

All on mt4s488 (122 steps) unless noted, LR 2e-6, eval 8k + 2k × 500 per task, compared against the dense curve
(dense 5e-6 at 31 / 61 / 122 steps, tdb campaign on Beaker) at matched `actual_pflops`. Launch with
`research/token_routing/online_router/launch_otr.py --arms <arm>`.

| arm | what it isolates |
|---|---|
| `otr30L4` (smoke) | input drop, within-document routing |
| `rnd30` | same budget, uniformly random tokens per row: the value of the router itself |
| `otr30M8` | mid-stack drop at L8, dropped tokens skip 24/32 blocks: the layer-skip lever |
| `otr30L4-ds05` | half the budget row-level: does cross-document allocation help, and what does it leak? |
| `otr30L4-gold` | + gold flag, doc_share 0.5: the gold-aware variant (veto if `leak_auc` → 1) |
| `otr15L4`, `otr50L4` | budget sweep for the Pareto curve |
| `otr30L4-de4` | a full-context step every 4 (mixing; distribution match to eval) |
| best arm on mt4s976 / mt4s1464 | the "more data at matched FLOPs" setting Prasann set on 2026-10-22 |
| best arm, 3 seeds | run-to-run noise is ±0.02-0.03 f1 at 8k (tokdrop_lr), so single-seed wins don't count |
| best arm on mt11s240, then 32k rows | the 11-task screen, then length generalisation |

**Decision rule.** Promote an arm only if it beats dense at matched PF on all four tasks at 8k by more than the
seed noise, with `leak_auc` near 0.5 (or no worse than the tier67 rule) and `rank_corr_explore` > 0 throughout.

## New levers, 2026-10-24 (code 076464355; all screens de0 -- no full-context steps, Prasann's rule)

| flag | what | FLOP / memory charge |
|---|---|---|
| `--st-otr-mod-layers 8-31:2` | mixture-of-depths: router picks keep_frac of each document at block 8; listed blocks run on those (+ non-body) only, the rest pass the residual; every token reaches the head | each listed block at the kept length; targets over every body token (uncensored) |
| `--st-otr-per-task-key-tokens 11` | one ridge router per task (key = the row's first 11 tokens = its prompt template; one key per mt4 / mt11 task file, nq + hotpotqa share one) | ~0.2 GB float64 Gram per task at F ~ 5150 |
| `--st-otr-freeze-schedule 4:0.25,7:0.5,12:0.75` | progressive freeze (grads + Adam state released per stage) | frozen below the drop: 1/3; frozen at/above it: 2/3 (activation grad still needed for the target) |
| `--st-otr-mass-target 0.8 --st-otr-max-keep-frac 0.2` | adaptive keep: per document the fewest top tokens covering 80% of predicted ‖g‖²; floor; per-row cap (peak memory) | realised keep |
| `--st-otr-target-layer 16` | fit on ‖dL/dh‖ at block 16: exploration tokens on routed steps (uniform, kept, uncensored); every token under MoD | hook only |
| `--st-otr-early-lora 16` | blocks [0, L) + embeddings frozen from step 0, rank-16 LoRA on every linear there (scale 32 = adapter LR multiplier); eval re-installs from config.json | blocks [0, L) at 2/3 + 6 r (in + out) per token |

Screens in flight (Lambda, mt4s1464, LR 5e-6, seed 0, vs otr10M7de0 2-seed mean 0.937 / 0.902 / 0.623 / 0.682 at
0.30x): otr10M7de0pt, mass80c20M7, lora16M7, prog4712M7, tl16M7, mod25L8. The per-task de16 run was cancelled under
the no-full-step rule; de32 / fa275de16 (launched before it) are references only: otr10M7de32 (0.32x) 0.927 / 0.906 /
0.616 / 0.558, fa275M7de16 (0.31x) 0.929 / 0.910 / 0.637 / 0.737 (seed 0 each, 8k, eval_size 500) -- neither beats the
de0 two-seed mean at 0.30x by more than seed noise on contra / nq.

**Screen results so far (8k contra / nq / oolong / outlier, eval_size 500, seed 0):**

| arm | PF (x dense) | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| otr10M7de0 shared router (2 seeds, reference) | 346 (0.30) | 0.937 | 0.902 | 0.623 | 0.682 |
| otr10M7de0pt (per-task routers) | 347 (0.30) | 0.941 | 0.890 | 0.616 | 0.773 |
| mod25L8 (MoD, capacity 0.25, blocks 8-30 alternate) | 905 (0.78) | 0.957 | 0.928 | 0.622 | **0.468** |
| mass80c20M7 (adaptive: 80% of predicted ‖g‖² per doc, row cap 20%) | 428 (0.37) | 0.941 | 0.918 | 0.585 | 0.571 |
| prog4712M7 (freeze [0,4) @25%, [0,7) @50%, [0,12) @75%), seeds 0 / 1 | 253 (0.22) | 0.933 / 0.919 | 0.890 / 0.906 | 0.620 / 0.647 | 0.508 / 0.602 |
| tl16M7 (router fit on ‖dL/dh‖ at block 16, exploration tokens) | 346 (0.30) | 0.919 | 0.910 | 0.628 | 0.726 |
| **lora16M7** (blocks 0-6 + embeddings frozen from step 0, rank-16 LoRA, scale 32), seeds 0 / 1 | 278 (0.24) | 0.939 / 0.909 | 0.872 / 0.892 | 0.615 / 0.595 | 0.566 / 0.592 |
| lora16x8M7 (rank 16, scale 8) | 278 (0.24) | 0.866 | 0.876 | 0.608 | 0.340 |
| lora16x128M7 (rank 16, scale 128) | 278 (0.24) | 0.929 | 0.882 | 0.620 | 0.523 |
| **lora64M7** (rank 64, scale 32; peak 48.4 GiB), seeds 0 / 1 | 286 (0.25) | 0.945 / 0.933 | 0.912 / 0.896 | 0.638 / 0.607 | 0.642 / 0.730 |
| lora128M7 (rank 128, scale 32) | 297 (0.26) | 0.919 | 0.910 | 0.588 | 0.703 |
| lora64k07M7 (rank 64, keep 7%) | 261 (0.23) | 0.929 | 0.892 | 0.620 | 0.650 |
| lora64k05M7 (rank 64, keep 5%) | 244 (0.21) | 0.912 | 0.900 | 0.609 | 0.534 |
| lora64M6 (rank 64, drop at block 6) | 270 (0.23) | 0.715 | 0.874 | 0.595 | 0.344 |
| lora64M5 (rank 64, drop at block 5) | 253 (0.22) | 0.752 | 0.882 | 0.605 | 0.381 |
| **lora64fa275M7** (rank 64, adapters + blocks 0-6 frozen at 75%), seeds 0 / 1 | 267 (0.23) | 0.931 / 0.937 | 0.898 / 0.904 | 0.608 / 0.636 | 0.756 / 0.717 |
| lora64fa183M7 (rank 64, frozen at 50%) | 248 (0.21) | 0.924 | 0.900 | 0.607 | 0.527 |
| dense-s1464 | 1154 (1.0) | 0.956 | 0.936 | 0.630 | 0.719 |

Per-task routing is neutral on the 4-task mix at matched PF (outlier +0.09 is inside that cell's 0.19 seed spread);
its case is the 11-task mix. MoD at capacity 0.25 is dense-level on contra / nq / oolong at 0.78x but loses outlier
(-0.25), and 0.78x is not an aggressive cut. Adaptive keep spends more (0.37x: the 80% mass cover keeps more than 10% of most documents) and is
no better than the fixed budget at less compute (de16, 0.34x: 0.946 / 0.912 / 0.625 / 0.725); oolong / outlier drop,
so the predicted-mass cover does not pick the tokens those tasks need -- not a win at this setting. Progressive freeze is the best contra of the freeze family at the lowest PF (0.933 at 0.22x vs fa183
0.924 at 0.24x and fa92 0.913 at 0.21x), level with no-freeze de0 (0.937, 0.30x) within seed noise; outlier 0.508 is
low but that cell's seed spread is 0.19. **Seed 1 (0.919 / 0.906 / 0.647 / 0.602) takes the 2-seed mean to
0.926 / 0.898 / 0.634 / 0.555 at 0.22x -- the same as single-step fa183 (0.924 / 0.900 / 0.609 / 0.657 at 0.24x) on
contra / nq, +0.025 oolong, -0.10 outlier: seed 0's contra lead was noise.** Progressive freezing saves 0.02x and ends at
41.5 GiB, but does not beat the single freeze. Fitting the router on block-16 gradients (tl16M7, exploration tokens only) is neutral at matched PF:
contra -0.018 (the de0 seeds differ by 0.021), nq / oolong / outlier +0.005-0.04 -- the depth drift of importance does
not show up as a better pick at this budget. **The LoRA early stack is the one lever that cuts PEAK memory: 48.1 GiB for the whole run vs 60.3 GiB for
de0 and for every warmup-then-freeze arm before its freeze (prog4712M7 peaks at 60.3, ends at 41.5).** At 0.24x it holds
contra (0.939 vs de0 0.937 at 0.30x), loses nq (-0.03) and outlier (-0.12, seed noise up to 0.19), seed 0 only. Its
evals re-installed the adapters (logged in all four; checkpoint-side rebuild, since this run's config.json predates
1f626f71d). **Seed 1 (0.909 / 0.892 / 0.595 / 0.592; adapters re-installed in all four evals) gives a 2-seed mean of 0.924 /
0.882 / 0.605 / 0.579 at 0.24x: the same quality as the single freeze fa183 (0.924 / 0.900 / 0.609 / 0.657, 0.24x)
within seed noise, at a 48 GiB instead of 60 GiB peak.** **LoRA sweep (seed 0): the scale only has to be large enough (8 loses 0.06 contra / 0.24 outlier; 32 and 128 are
level), and RANK matters -- rank 64 (0.945 / 0.912 / 0.638 / 0.642 at 0.25x, 48.4 GiB peak) is at or above the de0
reference (0.937 / 0.902 / 0.623 / 0.682 at 0.30x, 60.3 GiB) on contra / nq / oolong: about the same quality with 17%
less compute and 12 GiB less peak memory.** **Seed 1 (0.933 / 0.896 / 0.607 / 0.730; adapters re-installed in all four
evals, peak 48.4 GiB) confirms it: 2-seed mean 0.939 / 0.904 / 0.623 / 0.686 at 0.25x vs de0 0.937 / 0.902 / 0.623 /
0.682 at 0.30x -- identical quality on every task, 17% less compute, 12 GiB (20%) less peak memory from step 0.** Rank
saturates at 64: lora128M7 (0.919 / 0.910 / 0.588 / 0.703 at 0.26x, seed 0) is no better. Cutting the budget under LoRA r64 trades
smoothly but saves little: 10% -> 7% -> 5% keep is 0.25x -> 0.23x -> 0.21x for contra 0.939 -> 0.929 -> 0.912 and
outlier 0.686 -> 0.650 -> 0.534 (the 0.02-0.04x saved is small because blocks 0-6 still run on every token). LoRA at 5%
beats no-LoRA at 5% (otr05M7de0: 0.906 / 0.890 / 0.589 / 0.460 at 0.26x) on every task at less compute. **Dropping before block 7
still fails under LoRA r64 and LR 5e-6:** drop at 6 / 5 = contra 0.715 / 0.752, outlier 0.344 / 0.381, for 0.02-0.03x
saved -- block 7 (the first full-attention block after the GDN prefix) must see every token, as the split-point curve
showed at 2e-6. LoRA r64 + keep 10% at block 7 stays the point. **LoRA then freeze (5e1202e4b: the adapters freeze at
freeze_after, after which blocks 0-6 run forward-only) trades smoothly at the LoRA peak (48.4 GiB throughout):** no freeze
0.25x 0.939 / 0.904 / 0.623 / 0.686 (2 seeds) -> frozen at 75% 0.23x 0.931 / 0.898 / 0.608 / 0.756 -> at 50% 0.21x
0.924 / 0.900 / 0.607 / 0.527 (seed 0 each). Frozen at 50% equals the plain warmup-then-freeze fa183 (0.924 / 0.900 /
0.609 / 0.657 at 0.24x, 60 GiB peak) on contra / nq / oolong at less compute and 12 GiB less peak. (The fa275 oolong@2k
row was missing from the Lambda results copy -- the node-local eval had it, 0.858 on 500 -- re-copied and added.) **Seed 1 of
the 75% point (0.937 / 0.904 / 0.636 / 0.717) makes it the best compute + memory point: 2-seed mean 0.934 / 0.901 /
0.622 / 0.737 at 0.23x and 48.4 GiB, vs the de0 router 0.937 / 0.902 / 0.623 / 0.682 at 0.30x and 60.3 GiB -- the same
quality at 23% less compute and 20% less peak memory.** Running: lora64f16M7 and
lora16f16M7 on mt11s720 at 5e-6 (Beaker; evals from an eval worktree at the fixed tip). **LoRA eval trap fixed (1f626f71d, 984825862):** SoftToken defined
export_config twice, so the early_lora block never reached config.json and a DCP load silently drops adapter keys; the
eval hook now rebuilds the block from the checkpoint (or refuses) and guards every checkpoint key. Lambda evals run the
parent's checkout unless REPO_OVERRIDE is set (068d3f0ac); LoRA runs use OLMo-core-otr10.

**Parent's mt4s1952 winners re-run at mt4s1464 (LR 5e-6, de0, 8k, eval_size 500):**

| arm | PF (x dense) | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| otr10M7de0 (reference, 2 seeds) | 346 (0.30) | 0.937 | 0.902 | 0.623 | 0.682 |
| otr10M7de0sk3 (+ --train-skip-layers 20,24,28), seeds 0 / 1 | 333 (0.29) | 0.915 / 0.900 | 0.898 / 0.920 | 0.618 / 0.615 | 0.639 / 0.537 |
| **r20to7M7de0** (--st-otr-stages 7:0.2,16:0.07), seeds 0 / 1 | 355 (0.31) | 0.956 / 0.956 | 0.902 / 0.916 | 0.630 / 0.658 | 0.765 / 0.817 |
| lora64fa275 (reference: LoRA r64 + freeze 75%, 2 seeds) | 267 (0.23) | 0.934 | 0.901 | 0.622 | 0.737 |
| lora64fa275sk3M7 (+ sk3) | 254 (0.22) | 0.898 | 0.902 | 0.609 | 0.685 |
| lora64ptM7 (LoRA r64 + per-task routers) | 287 (0.25) | 0.932 | 0.882 | 0.620 | 0.686 |
| **lora64r20to7 (LoRA r64 + two-stage, NO freeze -- the universal recipe)**, seeds 0 / 1 | 296 (0.26) | 0.944 / 0.945 | 0.916 / 0.908 | 0.615 / 0.656 | 0.713 / 0.738 |
| **lora64fa275r20to7** (+ stages 7:0.2,16:0.07; peak 48.9 GiB), seeds 0 / 1 | 276 (0.24) | 0.941 / 0.935 | 0.928 / 0.916 | 0.593 / 0.626 | 0.811 / 0.718 |

sk3 2-seed mean 0.908 / 0.909 / 0.617 / 0.588: -0.03 contra and -0.09 outlier vs the plain router for 4% less compute
-- past block 7 only the kept 10% run the upper blocks, so skipping three of them saves little. (The parent's
s1952 result compared sk3 with the DENSE curve at matched PF, where it was within noise; against the same router it is
a small loss here.) **The two-stage drop r20to7 is the strongest router quality so far, confirmed on two seeds: mean 0.956 / 0.909 /
0.644 / 0.791 at 0.31x vs dense-s1464 0.956 / 0.936 / 0.630 / 0.719 -- equal contra, +0.014 oolong, +0.07 outlier, -0.027
nq at 31% of dense compute; vs the one-stage router at the same compute +0.019 / +0.007 / +0.021 / +0.11.** (Peak 60.8
GiB; on mt11s720 + floor 16 it does not beat the 0.36x arms.) **With the LoRA r64 + freeze-75% early stack (lora64fa275r20to7): 0.941 / 0.928 / 0.593 / 0.811 at 0.24x and
48.9 GiB** -- vs LoRA + freeze alone +0.007 contra, +0.027 nq, +0.07 outlier, -0.03 oolong, for +0.01x. sk3 on top of
LoRA costs contra (-0.036) for 0.01x; per-task routers on top of LoRA r64 are neutral (0.932 / 0.882 / 0.620 / 0.686 vs
0.939 / 0.904 / 0.623 / 0.686 shared, 0.25x). **Seed 1 (0.935 / 0.916 / 0.626 / 0.718; adapters re-installed in all four evals) gives a 2-seed mean of 0.938 / 0.922
/ 0.610 / 0.765 at 0.24x and 48.9 GiB: vs LoRA + freeze alone +0.004 contra, +0.021 nq, -0.012 oolong, +0.028 outlier for
+0.01x; vs the one-stage router (0.30x, 60.3 GiB) level on contra / oolong, better nq / outlier, at 80% of its compute and
81% of its peak memory.**

**Sweep around the combination (mt4s1464, LR 5e-6 unless noted, de0, seed 0; base 2-seed 0.938 / 0.922 / 0.610 / 0.765 at 0.24x):**

| variant | PF (x dense) | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| **stage-2 keep 10% (7:0.2,16:0.10)**, seeds 0 / 1 | 293 (0.25) | **0.956 / 0.943** | 0.916 / 0.920 | 0.620 / 0.647 | 0.759 / 0.781 |
| freeze at 85% (step 311) | 284 (0.25) | 0.953 | 0.916 | 0.618 | 0.730 |
| stage-1 keep 30% (7:0.3,16:0.07) | 303 (0.26) | 0.955 | 0.912 | 0.616 | 0.716 |
| **third stage (7:0.2,16:0.07,24:0.03)**, seeds 0 / 1 | 262 (0.23) | 0.943 / 0.951 | 0.914 / 0.918 | 0.622 / 0.622 | 0.759 / 0.761 |
| LoRA rank 128 | 286 (0.25) | 0.943 | 0.916 | 0.597 | 0.745 |
| LR 7.5e-6 | 276 (0.24) | 0.946 | 0.910 | 0.596 | 0.593 |
| stage-1 keep 15% (7:0.15,16:0.07) | 264 (0.23) | 0.943 | 0.912 | 0.594 | 0.665 |
| stage-2 keep 5% (7:0.2,16:0.05) | 263 (0.23) | 0.936 | 0.900 | 0.630 | 0.557 |

More budget anywhere (stage 2 to 10%, stage 1 to 30%, or a later freeze) lifts contra to dense level (0.953-0.956) for
+0.01-0.02x; cutting either stage loses outlier; a third stage to 3% at block 24 is level with the base at 0.01x less;
rank 128 and LR 7.5e-6 do not help. **Both winners hold on two seeds:** stage-2 10% = 0.950 / 0.918 / 0.634 / 0.770 at
0.25x (vs dense-s1464 0.956 / 0.936 / 0.630 / 0.719: within 0.02 on contra / nq, above on oolong / outlier, at a quarter of
dense compute and ~49 GiB peak); 3-stage = 0.947 / 0.916 / 0.622 / 0.760 at 0.23x (>= the base 0.938 / 0.922 / 0.610 /
0.765 at 0.24x). mt11s720 versions: stage-2-10% + floor in eval; LoRA + two-stage with no freeze / freeze 85% training.

**Universal recipe compute curve on mt4 (LR 5e-6, de0, 8k, eval_size 500; recipe s488 / s976 seed 0, s1464 2 seeds).**
Dense PF = steps x 3.17 PF (mt4 per-step dense cost); dense at 31 / 61 steps are the partial-epoch s488 runs (lrb).

| run | PF | contra | nq | oolong | outlier |
|---|---|---|---|---|---|
| **recipe mt4s488** | 103 | 0.815 | 0.890 | 0.547 | 0.381 |
| dense 31 steps | 98 | 0.639 | 0.800 | 0.500 | 0.209 |
| **recipe mt4s976** | 199 | 0.879 | 0.916 | 0.600 | 0.376 |
| dense 61 steps | 193 | 0.834 | 0.880 | 0.593 | 0.307 |
| **recipe mt4s1464** | 296 | 0.945 | 0.912 | 0.636 | 0.726 |
| dense mt4s488 (1 epoch) | 387 | 0.893 | 0.896 | 0.617 | 0.302 |
| dense mt4s976 | 773 | 0.931 | 0.916 | 0.584 | 0.432 |
| dense mt4s1464 | 1154 | 0.956 | 0.936 | 0.630 | 0.719 |

At matched FLOPs the recipe leads dense at every point (98-103 PF: +0.18 contra, +0.09 nq, +0.17 outlier; 193-199
PF: +0.045 / +0.036 / +0.007 / +0.07). FLOPs-to-match: dense reaches the recipe's s1464 contra (0.945) only between 773
and 1154 PF (~3.5x) and its outlier (0.726) at ~1154 PF (~3.9x). On the same rows the recipe is behind dense on contra
at s488 / s976 (it sees each row once at 0.26x the per-row cost) and level at s1464.

