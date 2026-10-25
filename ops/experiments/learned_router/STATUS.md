---
experiment: learned_router
title: End-to-end learned token-drop router -- now ONLINE at train time (ridge refit every step from the training backward; + mid-stack token exit), plus the earlier frozen-checkpoint routers and gradprobe
status: active
updated: 2026-10-25
branch: claude/online-router-1bc01a (online router, pushed); earlier: prasann/flopdrop_clean (gradprobe), prasann/landmark (frozen routers)
launcher: research/token_routing/online_router/launch_otr.py (online, Beaker); research/token_routing/learned_router/train_e2e_router.py (frozen, legacy)
records:
  - records/online-token-router.md
  - records/learned-token-router.md
  - records/devloss-grid-underperformers.md
  - records/gradprobe-token-importance.md
  - records/gradprobe-gradient-similarity.md
results:
  - research/token_routing/online_router/results/probe_base/train/offline-probe.json
  - research/token_routing/learned_router/gradprobe/results/json/
  - research/token_routing/learned_router/gradprobe/results/min_budget_vs_length_firm_tau0.1.png
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/layerskip/frontier_summary.txt
  - research/token_routing/learned_router/layerskip/frontier.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/layerskip/runs/nq/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/headline_shared.json
  - research/token_routing/learned_router/split_report_test64.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_router_t64/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_router_long/
artifacts:
  - https://claude.ai/artifact/BRAYbDr5uLW7KdZzeh2yL1
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/f8k-otr (online); https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/router-e2e (frozen)
next_step: ONLINE ROUTER (2026-10-25). Rule: no full-context steps (de0); cheap arms get more fresh rows. mt4s1464 LR 5e-6, 2 seeds each: (a) BEST QUALITY: two-stage drop --st-otr-stages 7:0.2,16:0.07 = 0.956/0.909/0.644/0.791 at 0.31x (dense 0.956/0.936/0.630/0.719; one-stage router 0.937/0.902/0.623/0.682 at 0.30x), 60.8 GiB. (b) BEST COMPUTE+MEMORY: LoRA r64 early stack + freeze at 75% = 0.934/0.901/0.622/0.737 at 0.23x, 48.4 GiB. (c) combined (LoRA r64 + freeze 75% + two-stage, 2 seeds): 0.938/0.922/0.610/0.765 at 0.24x, 48.9 GiB; sweep winners (2 seeds): stage-2 10% (7:0.2,16:0.10) 0.950/0.918/0.634/0.770 at 0.25x, 3-stage (+24:0.03) 0.947/0.916/0.622/0.760 at 0.23x. On mt11 freezing the LoRA stack costs xabsence early or late (fa75 0.544, fa85 0.584); NO freeze recovers it: LoRA r64 + two-stage + floor = 0.751 mean9 at 0.38x (2 seeds, +0.045 vs dense interp; = best mt11 arm at the LoRA memory peak). Stage-2 10% no freeze on mt11: 0.740. Universal recipe: LoRA r64 + 7:0.2,16:0.07, no freeze (mt4 0.26x, 2 seeds: 0.945/0.912/0.636/0.726, ~49 GiB). mt11s720 + floor 16 at 5e-6: freeze-half 0.750 (0.36x), LoRA r64 0.748 (0.37x), r20to7 0.748 (0.43x), sk3 0.735 (0.40x), vs dense interp ~0.70. Layer skip (sk3), MoD, adaptive mass, block-16 target, progressive freeze: neutral or worse (record). LoRA r64 + per-task neutral. Recipe compute curves (record): mt4 leads dense at matched PF at s488 / s976 / s1464 (FLOPs-to-match ~3.5x); mt11 s240-s960 leads by +0.04-0.05 mean9 (~1.7x). Recipe + per-task on mt11 neutral (2 seeds). IN FLIGHT: none of mine (parent: dense mt11 s240 at 32 / 64 steps for the s240 matched read). Code: claude/online-router-1bc01a (5e1202e4b+).
---

# learned_router

## Question
Can a learned per-token keep/drop router replace the dev-loss grid's fixed rules (`gold_fl20p8_noslot`
and similar)? The router is trained end to end on the frozen dense Qwen3.5-4B suite checkpoints, with no
heuristic init. A second question is how far a per-(token, layer) skip router on top of it can cut forward
FLOPs at ≤0.02/0.05/0.10 nats ΔCE.

## Decisions
- 2026-09-23 (agent): v1 was a linear router over [embedding, position, gold flag] trained with REINFORCE. It was bistable: keep-all or collapse. It beat the heuristics only on nq/outlier/msmarco.
- 2026-09-28 (agent): a differentiable relaxed router with a task-relative tolerance (§7) did not pass the gate on nq/outlier/contradiction, so it was not run on the full grid.
- 2026-09-29 (Prasann): the router must be END-TO-END. Never warm-start from or distill fixed rules such as gold_fl20p8: "optimal rules should emerge naturally". A neutral init plus a keep-all fallback is fine. Standing directive: `memory:router-end-to-end-no-heuristic-init`.
- 2026-09-30 (agent): canonical recipe v5/v6a.
  - `relpos_noemb` features, ρ matched to the bar, hard-concrete gates, CE+KL, 3 restarts plus polish.
  - The primary comparison is the paired per-row budget (`@pair`) against `gold_fl20p8_noslot`.
- 2026-10-01 (agent): stage 2 is a per-(token, layer) skip router on the frozen token router (`layerskip/`). Frontier tiers are 0.02/0.05/0.10 nats.
- 2026-10-02 (agent): recommended shared router = seed-averaged 6-task `xs6avg` (`weights/xs6avg/xs6avg_s0_rhobar.pt`). 17-task pooling is high-variance: `xt17v6a`'s 1/13/2 was a lucky seed.
- 2026-10-13 (Prasann): gradient sanity check for token dropping (gradprobe, record `gradprobe-token-importance`): per-task checkpoints, nq/outlier/contradiction, debug on 1 task / 2 rows first, 8k not 2k, token dropping only (layer skip deferred). Later (Prasann): keep gold as a baseline; prefer cheap gradient rules over IG if comparable; add setA SFT + setA's own pre-SFT base; 4-5 rows/cell acceptable, then firm up at 16 rows.
- 2026-10-13 (agent): one gradient pass at gate g=0.5 is the cheap importance rule (|grad| at g=1 is saturated noise on outlier/contradiction; 16-step IG no better at 8k). Adaptive g and iterative re-scoring dropped (no gain).
- 2026-10-14 (Prasann): setA via Beaker → abandoned (image); setA moved to Berkeley via S3 model-only copies. Firm-up on Lambda A100s when berkeleynlp's group cap was full (Prasann chose Lambda over evicting Rhys).
- 2026-10-16 (Prasann): close the gradprobe work for now.
- 2026-10-19 (Prasann → agent): base-model gradients, checkpoint drop+eval (`gradprobe/cross_score.py`, 64 rows/cell ⚠, nq/outlier/contradiction 8k, setAbase→setA32k and Qwen3.5-4B base→per-task). Base beats the checkpoint's saturated g=1 gradient on outlier/contradiction but never beats the checkpoint's own g=0.5 gradient (ties per-task nq/outlier; +0.09 to +0.72 worse elsewhere at the bar budget). Record §"Base-model gradients". A parallel session (grad-sim-drop) is measuring full-vs-dropped-context gradient similarity.
- 2026-10-19 (Prasann → agent, grad-sim-drop): full-context vs token-dropped TRAINING gradient (answer-CE parameter gradient, hard drop), on BASE checkpoints only (Prasann: SFT not needed; SFT'd setA32k/setAbase runs cancelled), Qwen3.5-4B-Base vs Qwen3-4B-Base (no GDN; nothink template), plus an exact per-token/per-layer decomposition (`gradprobe/grad_tokens.py`). Magnitude is kept (0.7-0.9x); the row-specific direction is mostly lost (centered cos 0.1-0.45); the cause is the incoming gradient δ at kept tokens being redistributed, not the dropped tokens' own share; no consistent GDN effect (the sign flips by task). Record `gradprobe-gradient-similarity`.
- 2026-10-20 (Prasann → agent, grad-sim-drop): same analysis on TASK-TRAINED checkpoints with the bar family (gold + 8 header tokens + first/last x%) as the drop rule. Prasann approved preempting Rhys on sneetches for this, with a cap of 8 jsteinhardt GPUs. Per-task SFT, setA32k and Qwen3-4B setA (lengen arm c, pulled weka→S3→sneetches as bf16 `/data/prasann/devloss_grid/ckpts/q3setA`), 48 rows each. Results:
  - The bar is lossless in CE but keeps only cos 0.26–0.76 of the training direction, and on rows that still carry loss it nearly zeroes the gradient (0.002–0.006×).
  - On outlier the bar matches ds64's frontier rules (ck64/flk64) in direction (0.2–0.33) with 40% fewer tokens.
  - No GDN effect at matched loss.
  - Dev-loss grid, Qwen3 setA vs setA32k: the bar is at parity on 24 vs 25 of 32 cells. Every SLOT scheme breaks on Qwen3 at 32k only (0–1 of 10 cells; undiagnosed); the no-slot schemes are fine.

## Online train-time router (2026-10-22, session online-router)
Brief (parent session, for Prasann's universal-rule goal): an ONLINE router that decides at train time which
tokens to drop (and optionally which layers they skip), learned during fine-tuning, so that training at a FLOP
budget beats dense at matched FLOPs. Design + numbers: `records/online-token-router.md`.
- Decision (agent): linear scorer over structure + lexical-query + hidden(h, h*q) [+ optional gold flag]
  features; ridge refit EVERY step on targets from that step's own backward (log ||dL/de_t|| at the compacted
  embedding output, per-row centred); step 1 keep-all (no heuristic init, per the end-to-end directive); 10%
  of the budget uniform-random exploration (selection bias + unbiased held-out metric).
- Decision (agent): `doc_share` = 1.0 default (budget spread over documents in proportion to length; router
  picks WHICH tokens of each document). Reason: gradient targets are gold-concentrated (gold AUC 0.83-0.94 on the
  base) and the parallel grad-sim report found the gradient keep rule leaks gold by count (AUC ~0.95), which is
  the known "dense = gold" shortcut. `online_router/leak_auc` is logged every step.
- Decision (agent): layer-skip half = mid-stack drop (`--st-otr-drop-layer L`): all tokens run blocks [0, L) on
  the full row, dropped tokens skip [L, 32); features free from the forward's own h^L. Not composed with the
  sibling layerskip-train session's whole-block layer drop (don't enable both).
- Every router FLOP (prefix pass, scoring, hook, ridge stats + solve, mid-stack per-block adjustment) is charged
  via `model._flop_extra_actual` -> `flop_meter/actual_pflops` (+ `flop_meter/overhead_pflops`).
- Tests: CPU 36 pass (horton, berkeleynlp normal); GPU 16 pass on Beaker H100 (01M5H9QQ5A1DNYDZTH7J7G3TS6;
  mid-stack rerun 01M5HA8S24PNN2VTZCFFSFMM8C: 79 passed, 0 skipped, incl. GDN-hybrid input_prefix + mid_stack modes, 2026-10-22). lorax CUDA init still broken (GPU tests skipped there).
- Offline probe (Beaker 01M5H9AW60673FJG0AXDGC6XMP; base markerfix, mt4s488, 48 fit / 32 held-out rows per task
  ⚠ eval_size=32): held-out Spearman of a pooled ridge probe vs log||dL/de||: structure 0.16-0.30, +lexical
  0.41-0.46, +embedding 0.44-0.58, +h4*q 0.51-0.61, +h8*q 0.54-0.63 (target ||dL/dh8||: 0.65-0.67, mass@30
  0.73-0.95 vs random 0.30); gold flag redundant given hidden features; compacted-row targets track full-context
  ones (Spearman 0.40-0.66). Recorded as `otrprobe_*` in ops results.
- Gradient fidelity (Beaker 01M5HBH3XHQ45E55RA7XZ3N77X, base, ⚠ 12 held-out rows/task): at keep 0.3 the MID-STACK
  drop (all tokens through blocks 0-7) keeps cos 0.88 / rel err 0.48 of the full parameter gradient vs 0.54 / 0.95
  dropping at the input; mid8 at keep 0.1 (≈0.36-0.40 of dense FLOPs) ≈ input at keep 0.5 (≈0.55). Router ≈ oracle
  under mid8 (0.88 vs 0.91); at the input the router beats random but not first_last (0.54 vs 0.59). Raw cos
  includes a shared direction (unrelated rows 0.40). Next: same probe on a TRAINED checkpoint.
- Launch bugs fixed 2026-10-22: wandb finish(quiet=) TypeError at shutdown (0b5b21ace; also killed the parent's tdb
  runs before export) and a trainer argparse check that rejected the online rule (c5f8106dc + a variant-wiring
  check, research/token_routing/online_router/check_variant_wiring.py). Arms relaunched on c5f8106dc.
- **Max train length at fixed 80 GB (A', 2026-10-25; 2 x A100, FSDP-2, full AC; `--mem-probe-steps`, 8b8a79d15).**
  Steady routed step: dense ~124k tokens; otr10M7de0 ~177k (1.43x); recipe ~205k (1.65x); LoRA-all + two-stage router
  ~266k (2.15x). Fixed state is 27.3 / 27.8 / 22.9 / 13.2 GiB; the transient is 0.386 vs ~0.25 GiB per 1k tokens.
  **Today all methods are capped at ~165k (1.33x)** by the dense passes, i.e. the trainer dry run and router warmup
  step 1; all OOM at step 0 at 196k / 256k. The routed peak is the recompute of full-length blocks 0-6, not the
  residual stream. Next (B'): route the dry run and warmup, then chunk the block 0-6 recompute. Record section
  "Memory vs train length"; raw output in research/token_routing/online_router/memprobe/.
  **B' round 1 (same day):** `--st-otr-warmup-random --skip-dry-run` (c5676a12f) remove both dense passes, and
  `--ffn-chunk-tokens 8192` (172f12f68, exact) chunks the FFN recompute. Max T is now: dense+chunk ~148k; recipe ~234k
  (1.58x; 196k fits at 70.3 GiB); LoRA-all ~300k (2.0x); otr10M7de0 ~210k. Next binding term: autograd C++ backward
  buffers, 22-32 GiB; probe them with stacks="all". Routed-warmup quality check: free (0.947 / 0.924 / 0.614 / 0.699
  at 291 PF vs the recipe's 0.945 / 0.912 / 0.636 / 0.726).
  **B' round 2:** those buffers were FLA's GDN backward rebuilding every chunk state at once.
  `--gdn-segment-tokens 16384` (a558feb2b; exact, GPU-tested) segments the GDN, carrying state between segments.
  **Max T with all four flags: dense ~165k; otr10M7de0 ~250k; recipe ~300k (1.8x); LoRA-all ~400k (2.4x).**
- **Adapter scope (B):** LoRA r64 on all 32 blocks at LR 1e-5 scores 0.946 / 0.908 / 0.625 / 0.719 at 261 PF, matching
  the recipe (296 PF) at 0.88x its compute (one seed). At 5e-6 it loses outlier (0.686). K=16 ≈ recipe; K=24 dips
  (likely noise). Rank 128 at K=32 gives no gain. 2 seeds at 1e-5: 0.940 / 0.913 / 0.623 / 0.703. The recipe at 1e-5
  loses oolong (0.565), so the K=7 recipe stays at 5e-6.
  **mt11s720: LoRA-all at LR 1e-5 scores mean9 0.755 at 232 PF, vs dense s240 at the same 232 PF: 0.697 (+0.058;
  outlier +0.14, oolong -0.02).** FLOPs-to-match ~2.2x (recipe ~1.7x). Running: seed 1 on mt11 (Beaker
  01M5SHHW6HX3TBEJSN1GSPPYHK); the mt4 compute curve s488 / s976 and LR 2e-5 (Lambda 114738-40).

## Current best numbers
All are teacher-forced answer-CE deltas on frozen dense 4B checkpoints. Nothing here is generation-graded.
⚠ eval_size is <500 everywhere, by design of the grid.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| token v6a + layer-skip, ≤0.05 tier | 16 tasks / 2k | mean per-task FLOP cut 4.48x (pooled x0.412 = 2.43x); meets on test 12/16 | 64/task ⚠ eval_size=64 | per-task | no-cot (TF CE) | debug/learned_router/layerskip/frontier_summary.txt |
| bar `gold_fl20p8_noslot` | 16 tasks / 2k | FLOPs x0.473, mean ΔCE +0.098 | 64/task ⚠ | — | no-cot (TF CE) | same |
| token f0.35 + layer keep 0.5 | nq / 2k | 10.24x fewer FLOPs, ΔCE −0.0015 | 64 ⚠ | 0.004 | no-cot (TF CE) | same |
| token f0.35 + layer keep 0.5 | outlier / 2k | 7.28x, ΔCE +0.021 | 64 ⚠ | 0.012 | no-cot (TF CE) | same |
| per-task v6a token router @pair | 16 tasks / 2k | 5 beat / 9 match / 2 lose (niah, outlier_amzn) vs bar | 64/task ⚠ | paired | no-cot (TF CE) | records/learned-token-router.md §9 |
| shared xs6avg @pair | 16 tasks / 2k / 8k / 32k | 12/16 at parity at 2k; 8k 1/7/7, 32k 1/5/9 (beat/match/lose) | 64 (2k), 48-56 (long) ⚠ | paired | no-cot (TF CE) | records/learned-token-router.md iters 65, 67 |
| token 0.5x bar (x0.099 FLOPs) | nq / 32k | ΔCE vs bar −0.0014; 10.3x wall-clock (CUDA graph, batch 1) | 56 ⚠ eval_size=56 | 0.0009 | no-cot (TF CE) | debug/learned_router/layerskip/runs/nq/longcheck_32k_L0.5_k0.5.json |
| + layer keep 0.75 (x0.078) | nq / 32k | 12.3x wall-clock vs full; ΔCE vs bar +0.000 | 56 ⚠ | 0.002 | no-cot (TF CE) | runs/nq/fastskip_32k_b1_k0.75.json (timing only) |
| + layer keep 0.5 (x0.054) | nq / 32k | ΔCE vs bar +0.110 (22/56 rows bad): does not transfer | 56 ⚠ | 0.015 | no-cot (TF CE) | longcheck_32k_L0.5_k0.5.json |

| **gradprobe** (oracle, uses the answer): gold first + \|grad\|@g=0.5, smallest lossless keep ratio (median ΔCE ≤ 0.1) | nq / 8k, 16k, 32k (setA) | 5% / 2% / 1% (≈330 doc tokens at each length) | 16 ⚠ eval_size=16 | bootstrap 10-90%: 2-5% / 2% / 1% | no-cot (TF CE) | ops results (`gradprobe_min_keep_ratio_goldga_tau0.1`); records/gradprobe-token-importance.md |
| gradprobe, same rule | contradiction / 8k, 16k, 32k (per-task) | 10% / 10% / 10% (771 / 1,582 / 3,214 tokens) | 16 ⚠ | bootstrap 5-10% | no-cot (TF CE) | same |
| gradprobe, same rule | outlier / 8k, 16k, 32k (setA) | 10% / 5% / 5% | 16 ⚠ | bootstrap ≤ 10% | no-cot (TF CE) | same |
| online-router TRAINING run-vs-run at 8k (contradiction / nq / oolong / outlier) | mt4 fresh data, LR 2e-6 vs dense 5e-6 | otr30M8-s1464 (619 PF): 0.951 / 0.918 / 0.633 / 0.615; otr10M8-s1464 (471 PF): 0.947 / 0.918 / 0.629 / 0.577; dense-s976 (≈773 PF): 0.931 / 0.916 / 0.584 / 0.432; dense-s1464 (≈1160 PF): 0.956 / 0.936 / 0.630 / 0.719 | 500 | ±0.01-0.02 per score, single seed | no-cot | ops results learned_router + tokdrop_lr |
| online-router TRAINING, Δ vs dense at matched PF (8k): contradiction / nq / oolong / outlier | mt4 fresh data s976 (244 steps), q35-4b markerfix, LR 2e-6 vs dense 5e-6 | mid-stack router otr30M8 (417 PF, 0.54x): +0.029 / +0.010 / -0.022 / +0.049; mid-stack random: -0.019 / +0.016 / -0.009 / +0.053; input router otr30L4 (348 PF): -0.100 / +0.004 / -0.019 / +0.046; input random: -0.265 / +0.002 / -0.021 / -0.023 | 500 | ±0.02-0.03 per diff (single seed) | no-cot | ops results learned_router; research/token_routing/online_router/results/matched_flop_table.md |
| online-router gradient fidelity on TRAINED otr30L4-s976, keep 0.3 | unseen mt4s1464 rows | mid-stack router 0.78 / 0.80, oracle 0.83, random 0.10 / 2.58; input router 0.45, first_last 0.28, random 0.06; unrelated rows 0.02 | 48 (12/task) ⚠ eval_size=48 | row medians | no-cot (TF grad) | ops results `gradfid_trained_*` |
| online-router gradient FIDELITY, cos(g_drop, g_full) / rel err, pooled 4 tasks, keep 0.3, per-document allocation | mt4 8k; q35-4b-base-markerfix | mid-stack drop @L8: router 0.88 / 0.48 (oracle 0.91 / 0.41, random 0.66 / 0.81); input drop: router 0.54 / 0.95 (first_last 0.59 / 0.86, random 0.46 / 0.97); unrelated rows 0.40 | 48 rows (12/task) ⚠ eval_size=48 | row-level medians | no-cot (TF grad) | ops results `gradfid_*`; records/online-token-router.md |
| online-router offline probe: Spearman(pred, log‖dL/de‖), pooled ridge, h⁸ + h⁸·q features | contradiction / nq / outlier / oolong, 8k; q35-4b-base-markerfix | +0.54 / +0.59 / +0.63 / +0.60 (structure only +0.30 / +0.25 / +0.26 / +0.16; random 0) | 32/task ⚠ eval_size=32 | row-level, not binomial | no-cot (TF grad) | ops results `otrprobe_*`; records/online-token-router.md |
| gradprobe: ΔCE at 10% kept, \|grad\|@g=1 vs @g=0.5 vs gold-first | outlier / 8k (per-task) | +0.68 vs +0.08 vs −0.03 (random +1.25) | 16 ⚠ | ~0.07-0.12 | no-cot (TF CE) | ops results (`gradprobe_dCE_keep10pct_*`) |
| same | contradiction / 8k (per-task) | +1.56 vs +0.33 vs +0.05 (random +1.68) | 16 ⚠ | ~0.06-0.12 | no-cot (TF CE) | same |
| gradsim: centered cos(g_full, g_drop), whole-model answer-CE gradient, random 10% / \|grad\|@0.5 keep 50% / bar | nq, outlier, contradiction / 8k; Qwen3.5-4B-Base | 0.11 / 0.42 / 0.27; 0.11 / 0.33 / 0.24; 0.21 / 0.21 / 0.25 (unrelated rows: −0.01 to +0.05) | 32 ⚠ eval_size=32 | row IQR ~±0.05 | no-cot (TF CE) | ops results (`gradsim_centered_cos_*`); records/gradprobe-gradient-similarity.md |
| same | same; Qwen3-4B-Base (full attention) | 0.11 / 0.42 / 0.26; 0.14 / 0.40 / 0.45; 0.26 / 0.35 / 0.35 | 32 ⚠ | same | no-cot (TF CE) | same |
| gradsim, task-trained: raw cos(g_full, g_drop) at the bar (x=20) / matched-k \|grad\|@0.5 | nq, outlier, contradiction / 8k; per-task SFT | 0.75 / 0.98; 0.26 / 0.50; 0.57 / 0.46 (unrelated rows ≈0.0–0.06) | 48 ⚠ eval_size=48 | row IQR ~±0.12 | no-cot (TF CE) | ops results `gradsim_taskckpt_*`; records/gradprobe-gradient-similarity.md §Task-trained |
| same | same; setA32k / Qwen3 setA | 0.61 / 0.76 (nq), 0.31 / 0.34 (outlier), 0.33 / 0.31 (contra) | 48 ⚠ | same | no-cot (TF CE) | same |
| outlier, bar vs ds64 frontier: raw cos | outlier / 8k; per-task SFT, setA32k | bar 0.26, 0.31; flk64 0.22, 0.33; ck64 0.21, 0.30 (keep 29% vs 48%) | 48 ⚠ | overlapping IQRs | no-cot (TF CE) | ops results `gradsim_outlier_cos_*` |
| dev-loss grid: cells at parity (ΔCE ≤ 0.02), bar `gold_fl20p8_noslot` | 11 setA tasks × 2k/8k/32k | setA32k 25/32, Qwen3 setA 24/32; slot `gold_fl20p8` at 32k: 9/10 vs 0/10 | 32–48 rows/cell ⚠ | paired | no-cot (TF CE) | ops results `grid_cells_parity002_*`; devloss_grid/results_family/setA32k_vs_q3setA.md |

## Next step
Length transfer is the main gap: 2k-trained routers lose most 32k cells, and layer keep 0.5 breaks at 32k.
Calibrate the layer cutoff on long val rows or train on mixed lengths. After that, check that the CE
parity holds under generation, in a trained model. Other open items from the record: niah needle cut on
some test64 rows, and textgroups at 32k needs doc-level features. The router weights (.pt) stay on local
storage and are not in git (`memory:flopdrop-clean-branch`).

gradprobe gradient-similarity + Qwen3 grid (closed for now, 2026-10-20; record `gradprobe-gradient-similarity`, artifact threads 02–04). Open follow-ups:
(a) diagnose the slot-at-32k break on Qwen3 (every `cent_cmean`-slot scheme 0–1/10 cells at 32k, no-slot fine; RoPE cache ruled out);
(b) check in training whether bar-dropped contexts starve the hard rows (gradient 0.002–0.006× on rows with CE ≥ 0.05; relevant to `tokdrop_lr` gfl20p8ns);
(c) a one-step SGD test of whether the partial direction (cos 0.3–0.75) matters for the full-context loss.

gradprobe (closed for now, 2026-10-16): the natural follow-ups are (1) a deployable router trained to imitate the
one-pass g=0.5 gradient (it is an oracle: it uses the answer), (2) a 64k point, which needs a memory-efficient
soft-keep attention (dense T×T bias OOMs at 66k tokens), (3) layer skipping with per-(token, layer) gates at g=0.5.

## Open issues / traps
- Every verdict uses eval_size 16-64 rows/task. The record says to read spreads and medians, not third decimals. Run-to-run variance (GPU nondeterminism) changes verdicts on single seeds, so require ≥3 seeds for subset claims.
- CE parity does not imply a right answer (`memory:ce-vs-chance-floor`). No generation-graded check of a routed construction exists yet.
- The xabsence column is VOID: the paraphrase-era checkpoint is scored on EXACT rows. niah, fiqa and absence select on only 16 val rows (`_stg`).
- Pooled training with `--xtask-offload` was 80% model swaps; fixed by pinned-memory parking (iteration 60).
- Layer skip converts to wall-clock only with fixed-capacity gathers plus CUDA graphs (`fastskip.py`). At 2k batch 1 it is latency-bound (2.8x vs 3.0x for token-only).
- At 32k the full-context model is itself +0.065 ± 0.030 worse than the bar on nq.
- gradprobe: per-row ΔCE curves have isolated spikes (+0.1 to +0.7 at one keep ratio): use the across-row median curve and τ = 0.1; mask nondeterminism alone moves ΔCE by 0.06-0.10. The per-task outlier 32k ρ* (30%) is unreliable (that model is weak at 32k; knee ≈2% by eye).
- gradprobe: setA's pre-SFT base ignores the context (any dropping lowers CE), so it cannot test the "|grad| is saturated" story.
- gradsim: on saturated SFT checkpoints, mean-subtraction centering (`--demean 1`) is meaningless (per-row norms differ up to 1000×, gives ≈0.99 for any rule). Use `--demean 2` (projection) or raw cosine (unrelated rows ≈0 there).
- Qwen3 answer-CE needs the nothink tokenizer (stock template puts `<think></think>` in the loss span): `/data/prasann/hf_tokenizers/Qwen3-4B-nothink` (mooney) or the training copy `/data/prasann/devloss_grid/ckpts/q3setA_tokenizer` (sneetches). `memory:qwen3-stock-template-think-in-answer-mask`.
- Dev-loss grid on Qwen3: slot schemes break at 32k only (see Next step (a)); use `_noslot` schemes on full-attention models.
- Staged for this investigation: sneetches `/data/prasann/devloss_grid/ckpts/q3setA` (bf16, 8.8 GB) + `q3setA_tokenizer`; mooney ckpts `setA32k`, `setAbase`, `q35-4b-base-modelonly`, `qwen3-4b-base-trainedmark`, `ctc-4b-{nq,outlier,contradiction}-full`; S3 `_transfer/gradprobe/ckpts/q3setA` (17.6 GB fp32 + tokenizer). sneetches `/data` was 100% full (28 GB free) on 2026-10-20.
- gradprobe code is UNCOMMITTED on prasann/flopdrop_clean (`research/token_routing/learned_router/gradprobe/`, `records/gradprobe-token-importance.md`). Staged inputs still on disk: horton/sneetches `/data/prasann/devloss_grid/{ckpts/setA32k,ckpts/setAbase,data/*/rung_16384,rung_65536}`, lorax `/data/prasann/devloss_grid/ckpts/ctc-4b-nq-full`, thidwick `/data/prasann/devloss_grid/ckpts/q35-4b-base-modelonly`, Lambda hyperplane02/05 `/tmp/prasann/gp` (≈19 / 37 GB), S3 `_transfer/gradprobe/` (code tarballs + setA copies, 36 GB).

## Verified vs unverified
- Verified 2026-10-12 from `layerskip/frontier_summary.txt`: the tier means (≤0.05: x0.412 pooled, 4.48x mean per-task, 12/16 meets), nq 10.24x, outlier 7.28x, and the bar x0.473 / +0.098.
- Verified from `layerskip/runs/nq/longcheck_32k_L0.5_k0.5.json`: full +0.065, token 0.5x bar at x0.099 −0.0014 ± 0.0009, and +layer keep 0.5 +0.110 ± 0.015 on 22 of 56 rows. `fastskip_32k_b1_k0.75.json` gives 12.27x graphed vs full at x0.078.
- UNVERIFIED: the quality of layer keep 0.75 at 32k (+0.000 ± 0.002). The record cites `runs/nq/longcheck_32k_L0.5_k0.75.json`, but no such file exists under `debug/learned_router/`; only `_k0.5` does.
- UNVERIFIED: the per-task 5/9/2 and xs6avg 12/16 tallies. They were read from the record's iteration log and not recomputed.
- UNVERIFIED — confirm with Prasann: is this still active on `prasann/flopdrop_clean`? There has been no commit since ec8242072 (2026-10-02).
- Verified 2026-10-20 (gradsim / Qwen3 grid): every number in the gradsim and grid rows above was recomputed from the run JSONs (sneetches `/data/prasann/gradprobe/out/gsim{bf,flk}_*`, `/data/prasann/devloss_grid/results_{setA32k,q3setA}/`; mooney `/data/prasann/gradprobe/out/{gsim,gtok}_*`) and recorded with `ops result add --check`. Flags: eval_size<500; some rows missing ckpt_path/eval_job_id (run ids carry the job ids). Checks passed: rerun cos ≥ 0.9999, keep-all cos ≥ 0.9999 with ΔCE 0, bar mask matched mask-for-mask, and Qwen3 training tokenizer prefix parity.
- UNVERIFIED: ds64's outlier frontier (`ck64`/`flk64`) taken from ds64 STATUS/launch files. The `results/ds64/results.csv` they cite is not on disk.
- Verified 2026-10-16 (gradprobe): every gradprobe number above recomputed from `gradprobe/results/json/` and recorded with `ops result add --check` (flags: eval_size<500, no eval_job_id for multi-run aggregates). Per-row parity checks passed in every run (soft g=1 vs flash ≤ 3e-2 on base, ≤ 1e-2 elsewhere; soft binary vs hard ≤ 3e-2; checkpointed vs plain gradients rel 2.6e-3 = rerun noise; olmo_st copy reproduces the distcp CE; Lambda A100 reproduces horton on the same row; bar mask reproduced token-for-token).
