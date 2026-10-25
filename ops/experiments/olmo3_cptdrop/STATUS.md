---
experiment: olmo3_cptdrop
title: Token-dropping CPT screen on OLMo 3 7B — stage2 → 32k on the Dolma 3 Longmino pool; dense vs length curriculum vs input-drop (random-20% blocks, first+last 25%), full-context dev CE at 3 matched FLOP budgets
status: active
updated: 2026-10-22
branch: prasann/flopdrop_clean (uncommitted: research/token_routing/olmo3_cptdrop/)
launcher: research/token_routing/olmo3_cptdrop/
records:
  - records/olmo3-cptdrop-screen-plan.md
results:
artifacts:
  - https://claude.ai/artifact/4tZKEXtpSpg5ax6B1YeoLk
wandb_group:
next_step: cubbins 4079442 (r128 K16, r64 K24, r64 K16 s1) eval -> 4106229 SHORT (S4 4k rows 12-960 steps; S4 -> routed-32k candidate / dense-32k control; LMIX packed controls). Waiting on router session: --st-otr-route-min-len (hybrid, per document in packed instances) and --ffn-width-frac (partial MLP). Data: data/mix4k_r0 (tail rows 39160-42999, 30,720 4k windows), mix4k_r25 / r50 (rows 0-3999) on cubbins.
---

# olmo3_cptdrop

## Question
Starting from a base that has never seen long context (OLMo 3 7B stage 2, 8k), does input-level token
dropping during long-context CPT beat dense at every matched FLOP budget — not only at the cheap end, where
softdetach won on an already-long-trained Qwen3.5-4B before dense overtook it near 2k PF?

## Decisions
- 2026-10-24 late (Prasann, via the LR-sweep session): 0.81-0.85x is not enough -- he wants BIG FLOP wins on CPT, and
  "something that looks like token dropping to learn the long range dependencies"; memory metric = max trainable
  context length at a fixed memory budget (not peak). Levers queued: short-window CPT (4k rows, 1/8 tokens/step) with a
  routed 32k phase (candidate) or dense 32k phase (control); packed length-mix control; hybrid (short dense + routed
  long rows in one step) once the router session adds route-min-len; partial-MLP (--ffn-width-frac) once it lands.
  Dense full-context steps mixed into a method = control only, never the candidate.
- 2026-10-24 (Prasann, via the LR-sweep session): (1) keep 32k CPT, no 64k move; try cuts that keep every token's
  label through the full depth (freeze bottom of stack after warmup, progressive freeze, mixture-of-depths).
  (2) NO dense / full-context steps mixed in at all (--st-otr-dense-every 0; no dense-keep-frac arms). (3) Cheap arms
  are compared at matched FLOPs and FLOPs-to-match-dense on MORE FRESH rows, never repeated epochs.
- 2026-10-20 (Prasann): CPT from a raw OLMo 3 checkpoint, longmino pool as data.
- 2026-10-21 (Prasann): no frozen dev-loss screen (may not correlate with training value); an empirical
  screen of strategies on short training runs instead.
- 2026-10-21 (Prasann): arms D (dense), C (8k → 32k curriculum), I-rand, I-fl only; token exit and layer
  skipping deferred. Keep the H200-hour count low (plan ≈73 H200-h).
- 2026-10-21 (Prasann): 32k first, 64k later.
- 2026-10-21 (Prasann): dense uses the OLMo 3 stage-3 optimizer recipe as is (peak LR 2.0712e-4); drop arms
  may use a different LR. Agent default: same LR for drop arms, warmup scaled to 1.7% of the run.
- 2026-10-21 (Prasann): Beaker and Lambda are both allowed where useful. Agent: S3 is the hub
  (`s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/`).
- 2026-10-21 (agent): start = `stage2-step47684` (the state OLMo's own long-context stage started from);
  stage3-step11921 converted too, as a reference for the eval sanity check. Blocks are positional (no
  marker tokens, so no marker-embedding repair). Dev from the 2^17+ buckets (not in the 7B stage-3 mix).
- 2026-10-21 (agent): build node mooney — cubbins is stuck in `completing` (other users' jobs hung in epilog).
- 2026-10-21 (agent): trainer flags added to `recipes/train/research_trainer/train_ctc_suite.py` (uncommitted):
  `--weight-decay`, `--z-loss`, `--warmup-fraction`, `--scheduler {linear,wsd}`, `--wsd-decay-fraction`,
  `--branch-from` (resume model+optim+trainer state = decay branch); `--rope-yarn-factor` is passed INTO
  `olmo3_7B_ctc_swa(yarn_factor=...)` (olmo3_configs.py) instead of stacking a second with_rope_scaling.
  Verified on the built config: x4 on the 8 full-attention layers (3,7,...,31), SWA layers plain RoPE,
  default unchanged at x8. Budgets as steps 120 / 240 / 480 (16 rows x 32k), `--save-interval 108`
  gives the 0.9 x budget branch points 108 / 216 / 432.
- 2026-10-21 (Prasann): arm C runs separately per budget (60M / 125M / 250M), the three in parallel. Recipe
  per budget B steps (512k tokens/step both phases): phase 1 = `--data data/c8k --seq-len 8192
  --global-batch 64 --scheduler wsd --warmup-fraction 0.017 --max-tokens B*524288 --max-steps B/2`
  (stable LR, full checkpoint at the hard stop); phase 2 = `--data data/train32k --seq-len 32768
  --global-batch 16 --branch-from <phase1>/step{B/2} --branch-optim-only --scheduler wsd --warmup-fraction 0
  --wsd-decay-fraction 0.2 --max-tokens (B/2)*524288` (decay = last 10% of B). `data/c8k` = train32k rows
  39,160-42,999 cut into 15,360 x 8,192 (`build_c8k.py`); phase 2 reads from row 0, so phases share no text.
- 2026-10-21 (session positional-512-token-blo, commit 2a0d30ef8 on branch claude/focused-lovelace-abac0f, not
  merged): positional blocks for the input-drop path. Arm flags: I-rand = `--variant softtoken --st-block-size 512
  --st-gold-blind --st-drop-slots --st-keep-prob 0.2` (kept 0.2007, sd 0.048 per row); I-fl = same with
  `--st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25` (exactly 0.25: first 64 + last 64
  per 512-block). Also fixed: random block draw identical on every positional row (now hashed with row tokens),
  and the per-example FLOP meter charging every EOS-led row as length 1 (affected dense too). Adds
  `flop_meter/kept_label_frac`; logged train CE on drop arms = true CE x kept fraction (divisor counts labels
  before dropping). Pass `--skip-marker-audit` (shards have no markers).
- 2026-10-21 (Prasann): OK to preempt Rhys on ONE node, the shortest-running: cubbins (his `preemptive` RL job
  4012765, 6 min old; sneetches' had run 5.5 h). mooney/mcfuzz are `preemptive_high` (danat / Rhys), not preemptible.
- 2026-10-21 (Prasann): dev set to 1,024 rows (paired SE ~0.006/bin at 256); short guard rebuilt from 7 stage-3 shards.
- 2026-10-21 (agent): merged into prasann/flopdrop_clean (local, not pushed): d81c4ab2d (experiment files),
  86d5c328d (o3cd/swa-position-window), a7364ca3a (o3cd/positional-blocks; conflicts in variants.py / model.py were
  two independent new kwargs, both kept), f022f00c3 (o3cd/devce-eval files + its ctc_devloss_grid hunk; another
  session's uncommitted edit to that file left uncommitted). Launcher run_arm.sbatch + driver_dense.sbatch (d59e1ac8b).
  All arms run --no-compile (compacted lengths vary per step) so wall-clock stays comparable.
- 2026-10-21 (agent): matched-FLOP budgets from the smoke FLOP meter (actual/dense per step): IRAND 0.184 ->
  650 / 1,300 / 2,600 steps (save every 585); IFL 0.223 -> 540 / 1,080 / 2,160 (save every 486); C phase-1 8k
  0.89 -> C at equal tokens 120 / 240 / 480 (~0.945x dense FLOPs). m is read from ACTUAL FLOPs on the dense
  curve, so budgets need only be close. Dense FLOP meter: 36.6 PF per 524,288-token step.
- 2026-10-22 (Prasann): run overnight, never wait long, use all compute, test small first, goal = a curve that is
  Pareto-better than dense at every budget; token/layer skipping may use is-gold (SFT only), position within the
  document and document length. Agent: (1) handoff arms HO (drop -> dense) as the 'envelope' candidate, branched
  from IRAND checkpoints; HO FLOPs = IRAND part (S0 x 0.184 x 36.6 PF) + the HO run's own meter; (2) background
  agent porting token exit (layer skipping for dropped tokens) + position-aware keep rules (recent window, ramp)
  in a separate worktree, for the next small-budget screen.
- 2026-10-22 (Prasann): TARGET = 10-20x FLOP Pareto-optimality. Minimum bar = the earlier positive Pareto results
  (softdetach sd20 ~4x at <=1.5k PF on Qwen3.5-4B 64k; ds64 soft tokens 2.2-4x on 4 tasks). Longer context allowed.
  Agent: (a) aggressive keep (I-fl 10% / 5%) at TINY budgets (dense-equivalent 12/24/48 steps) against dense
  120/240/480 + a measured dense low end (DLOW: 24, 48) so m=10-20 is read off measured points, not
  extrapolated; (b) a 64k screen (train64k from 2^16 docs, dev64k_512 from 2^17+, YaRN x8, 8 rows/step), since
  attention's share grows with length; (c) budgets for new keep rates use estimated FLOP ratios (~0.09 at 10%,
  ~0.046 at 5%); m always uses the measured actual_pflops.
- 2026-10-22 (agent): token exit + position-aware keep rules (background agent, commit 00f302d90, kept as branch
  o3cd/tokexit-posrules) merged as a7855efa8; flop_meter conflicted with another session's layer-drop feature:
  both kept, token exit + layer drop in one step refused by the meter. Arms in run_arm.sbatch: IFL_R4K
  (--st-keep-recent 4096), IFL_RAMP (--st-keep-ramp 0.1,0.5), IFL_X4/X8, IRAND_X4/X8 (--tokexit-layer).
  Measured meter actual/dense on synthetic 32k rows: I-rand 0.176, I-fl 0.223, recent4k+I-fl 0.312,
  ramp 0.271, I-fl L4 0.315 / L8 0.408, I-rand L4 0.274 / L8 0.373. Token exit's full-row prefix caps its
  saving near 32/L, so it cannot reach 10-20x on FLOPs alone; screen it only as a learning-per-FLOP probe.
  GPU tests (job 4017524, berkeleynlp normal) still queued; doc-position (window/doc_tokens) features not built.
- 2026-10-22 (Prasann): use Beaker as well ("jobs may take longer to allocate, keep an eye on that"). Agent: C120 /
  C240 / C480 / IFL on Beaker in parallel, IRAND on cubbins after dense. Pushed prasann/flopdrop_clean (e81d2c747;
  the branch was otherwise already pushed by another session).
- 2026-10-22 (agent, deviation from 'dense LR as is', reported to Prasann): LR probe (cubbins 4017622) at 120 / 240
  steps: peak LR 2e-5 beats the OLMo recipe's 2.07e-4 by -0.133 +-0.002 nats at 120 steps (paired) and 6.5e-5
  by -0.026 +-0.0003; at 2e-5 dense still improves 120 -> 240 (-0.0026 +-0.0001) and short-document CE barely
  moves (1.161 / 1.165 vs 1.156 at the start; 2.07e-4 forgot to 1.33-1.49). DECISION: peak LR 2e-5 for dense
  AND every drop arm. All old-LR Beaker jobs cancelled and relaunched at LR=2e-5 DIRTAG=lr20 (bk11, code
  40925312f); cubbins driver_lr20.sbatch (4017693). Most of the gain (stage2 -> 120 steps: -0.162) arrives inside
  the first 120 steps, so the 10-20x question lives at dense-equivalent 12 / 24 / 48 steps: measured there.
- 2026-10-21 (agent): converter's built-in validation is skipped in favour of `check_conversion.py`
  (fp32 + bf16 parity incl. a 6,144-token row so the sliding window is exercised).

## Runs
| what | id | state |
|---|---|---|
| builder smoke (200 train / 22 dev / 16 short rows) | slurm 4012901 + 4012903 (mooney) | PASSED after a User-Agent fix for olmo-data.org |
| data build train32k / dev32k / short4k | slurm 4012904 (mooney) | DONE 19.5 min: train 43,000 x 32,768 = 1.409B tokens (32,382 docs); dev 256 rows (23 topics, 0 doc overlap with train); short 256 x 4,096 |
| convert stage2 HF -> OLMo-core | slurm 4012907 (mooney) | checkpoint SAVED; converter's bf16 self-check "failed" (atol 1e-4 vs bf16 noise). fp32 parity (4012938) PASS: max logit diff 4e-5 (random 512) / 1.4e-3 (real 6,144, SWA exercised), CE identical; bf16 CE diff 3e-4. 4012906 failed earlier: converter needs a local HF dir |
| convert stage3 + parity | slurm 4012946 (mooney) | DONE: fp32 parity PASS (max logit diff 4e-5 random / 4e-4 real 6,144, CE identical; bf16 CE diff 3e-5) |
| upload c8k to S3 | slurm 4012971 (mooney) | DONE |
| dev32k_1024 + short4k_v2 rebuild | slurm 4015937 (mooney, CPU) | RUNNING (4015752 failed: a malformed JSON line in a pool file; now skipped and counted) |
| S3 -> cubbins: data + stage2 base | slurm 4016138 (cubbins, CPU) | RUNNING |
| dense driver 4016185 / 4016562 / 4016652 / 4016772 | cubbins | FAILED or CANCELLED in turn: base path not the model_and_optim subdir; SMOKE mode disabled checkpointing; 0-step WSD decay in a 5-step smoke; c8k empty-epoch livelock (80 min). Fixed in turn (f0bef5564, 1bb94f267, ae477fd8b, 5622d24ac) |
| dense driver 4017418 (code 5622d24ac) | cubbins 8xH200 | SMOKE PASSED (all six; earlier attempt passed pytest 85/85, D, decay branch from step3, C2 optim-only). RUNNING o3cd-D-480 (12.3 s/step, ETA ~01:20 10-22) -> o3cd-D-120 / o3cd-D-240 branches -> o3cd-D-120-s1 |
| rest driver 4017424 | cubbins | CANCELLED before start: IFL + C moved to Beaker |
| rest driver 4017448 | cubbins | CANCELLED before start: replaced by 4017452 |
| rest driver 4017452 (code b91b22b93) | cubbins, after 4017418 | QUEUED: EVALD (8-GPU row-chunked dev CE: stage2 + D-120/240/480/120-s1) -> o3cd-IRAND-2600 (saves every 195; +650, +1300 branches) -> HO-120/240/480 (IRAND 390/585/1170 steps then dense to the budget, trainer state kept) -> EVALI |
| Beaker C120 / C240 / C480 / IFL | 01M5H7ZSCJJ9HCWS89SMXTK8MB / 01M5H7ZYHY2YNE0K0910AJR16J / 01M5H80459Q9A5DW6X1SR2HS5P / 01M5H80948ZEMYGSAATK1Z8VM2 (jupiter 8xH100, urgent, minRuntime 1h, code e81d2c747) | PENDING since ~00:20 10-22. Each: S3 -> weka `.../prasanns/olmo3_cptdrop/beaker_<FAM>`, run_arm recipe, in-job dev CE on dev32k_1024 + short4k_v2, eval JSONs + exports -> S3 `devce/`, `runs/` |
| Beaker GEN wave 2 (32k, code 66ded0975) | DLOW 01M5H8QY86S1D371P8MWRDTCXR / IFL10 01M5H8R3573J2CPPG9BDACYM22 / IFL05 01M5H8R89Y432STWCVTACJQW1A | PENDING: DLOW = D-48 (+D-24 branch); IFL10 = keep 0.10, 533 steps (+133, +267); IFL05 = keep 0.05, 1,040 steps (+260, +520). First launch of the three (01M5H8P...) cancelled: a concurrent push race made gantry's ref uncertain |
| Beaker attempts 1-3 | C/IFL/GEN ids 01M5H7Z.../01M5H8Q.../01M5H9F... | all FAILED or CANCELLED: image has no boto3 (C240 died in 5 s); then gantry --install true = empty venv (no torch). Fixed in 0b16ebdba: --no-python + /opt/conda/bin/python + missing light deps; CPU CHECK job 01M5H9SCR80NCBWGKZ325STGGA passed |
| Beaker attempt 4 (code 0b16ebdba) | C120 01M5H9SW0JE7RZPT0AR2RJWYMP, C240 01M5H9T1DEJTWY580K5K6XP86F, C480 01M5H9T6FPZ943TECXW2H5JWAC, IFL 01M5H9TC39E64ZHNKW4STAK997, DLOW 01M5H9THF152NHBXCYQ5MSAENQ, IFL10 01M5H9TPQFEKP5QT39C9ZRHSZH, IFL05 01M5H9TWK5D5D3EEN8K649MPCF | PENDING since ~01:00 10-22 (ops rows o3cd-bk3-*) |
| data train64k + dev64k_512 | slurm 4017455 (mooney, CPU) | DONE 25 min: train64k 20,000 x 65,536 (1.31B tokens; doc p10/p50/p90 68k/85k/117k), dev64k_512 512 rows (2^17+). On S3 |
| Beaker 64k screen (code 0b16ebdba, YaRN x8, 8 rows/step) | D64 01M5HA5K5BN4K6EWRVBGNDMBWC (480 +120,+240), D64LOW 01M5HA5T0QNTF7E6FXARAS9Q79 (48 +24), IFL10x64 01M5HA62V26HZ3EAR8S7AZ5Z38 (658 +164,+329), IFL05x64 01M5HA67X8S89DR32XYQF1F1Q0 (1333 +333,+667), IFL25x64 01M5HA6FPA82AGH9BBQYBETY0V (1200 +600) | PENDING since ~01:10. Budgets from ESTIMATED 64k FLOP ratios (~0.073 at 10%, ~0.036 at 5%); m uses measured actual_pflops |
| dense driver 4017418 | cubbins | o3cd-D-480 DONE (480 steps, 1 h 43 m; checkpoints 216/324/432/480 kept, 108 pruned); D-120 branch FAILED (step108 gone) -> driver ended |
| rest driver 4017452 | cubbins | CANCELLED 2 min in (IRAND would have lost branch points) |
| rest driver 4017567 (code f1cc054c6) | cubbins | RUNNING/QUEUED: DFIX (D-240 from step216, D-120 fresh s0, D-120-s1) -> EVALD -> IRAND-2600 (all ckpts kept) -> HO-120/240/480 -> EVALI |
| Beaker relaunch (code 3e590babf, own weka dir per job) | IFL 01M5HBA9128MWQ1RS3G1DRNSM3, IFL10 01M5HBAE4TNGZG938G5HBEVNC4, IFL05 01M5HBAKJB8VZX8G6979WK733Q, D64 01M5HBASMD94ERBHD378BR9HB7, IFL10x64 01M5HBAZ0H7E581ED55NK8568J, IFL05x64 01M5HBB4518X6FB61QZ604GTYY, IFL25x64 01M5HBBAKYWAHGE5HXN502X6TB | PENDING (previous launches cancelled before start for the max_checkpoints trap). Still running from attempt 4: C120 / C240 / C480; DLOW pending on 0b16ebdba (only 2 saves, unaffected) |
| dev32k_1024 + short4k_v2 | mooney 4015937 | DONE: 1,024 dev rows (1 malformed pool line skipped); short guard 256 rows from 354,747 short docs over 7 stage-3 shards. Uploading to S3 (4017421) |
| upload data + both bases to S3 hub (`s3_upload.sbatch`) | slurm 4012955 (mooney) | DONE 5.5 min: data/{train32k,dev32k,short4k,c8k} + base/{stage2-step47684,stage3-step11921} at s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/ (4012954 failed at once: `--wrap` runs sh) |
| LR probe driver (cubbins) | slurm 4017622 | DONE: Dlr65-240 (+120 branch), Dlr20-240 (+120 branch), stage3-yarn8 rescored on dev32k_1024 |
| Beaker old-LR waves bk3-bk10 | (ops rows o3cd-bk*-*) | CANCELLED 2026-10-22 ~05:25 while pending: superseded by LR 2e-5 |
| LR-2e-5 driver (cubbins, code 40925312f) | slurm 4017693 | RUNNING from 05:30: DC (Dlr20-12/24/48, Dlr10-120) -> EV -> IR (IRANDlr20-2600, +650/+1300) -> HO (HOlr20-120/240/480) -> D480 (Dlr20-480) -> EV |
| Beaker bk11 (LR 2e-5, code 40925312f, weka beaker_<NAME>_lr20) | C120 01M5HSDVWGB5HABHZVH2H7YH4C, C240 01M5HSE1W2WMJHRH65V18V5QM8, C480 01M5HSE7BNP0Z7JR80JBVC10J6, IFL 01M5HSECB5PQ16CNA3B9MVZ64J, IFL10 01M5HSEHH1PWTATYBC5RNFYRCV, IFL05 01M5HSEPCT5ATWAZ1E4MBRXE33, D64lr20 01M5HSEW1E99JNJPFXAT1GXMD3, IFL10x64 01M5HSF114C1WVDGYW6B8Q5SZ6, IFL05x64 01M5HSF5KWPNYKF5WDR9VQFMSF, IFL25x64 01M5HSF9VY0MV1ETJB0CE4NGN1 | PENDING since 05:26 |
| dev-CE sanity (`eval_devce.py`, branch `claude/vigorous-agnesi-20ecb0` @ a151f921d): stage2 none/x4, stage3 x8 on dev32k + short4k | slurm 4012959 (mooney, 1 H200) | DONE: 6 CPU tests pass; per checkpoint load ~30 s + dev32k 372 s (1.45 s/row) + short4k 40 s; JSONs `/data/prasann/olmo3_cptdrop/devce/` (mooney) |

## Current best numbers
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| stage2 YaRN x4 (start) | dev32k_1024 / 32k | CE 1.8477 (16-32k 1.8400); short4k_v2 1.1559 | 1024 rows (short 256 ⚠) | paired below | no-cot | cubbins /data/prasann/olmo3_cptdrop/devce/ |
| D-120 (OLMo LR 2.07e-4) | dev32k_1024 / 32k | CE 1.8185 (16-32k 1.8153); short 1.3270 | 1024 | vs stage2 -0.0291 ±0.0053 | no-cot | same |
| D-120-s1 | dev32k_1024 / 32k | CE 1.8096 (16-32k 1.8068); short 1.3260 | 1024 | vs D-120 -0.0090 ±0.0011 (seed noise) | no-cot | same |
| D-240 | dev32k_1024 / 32k | CE 1.8290 (16-32k 1.8248); short 1.3854 | 1024 | vs D-120 +0.0105 ±0.0011 | no-cot | same |
| D-480 | dev32k_1024 / 32k | CE 1.8237 (16-32k 1.8206); short 1.4917 | 1024 | vs D-120 +0.0052 ±0.0012 | no-cot | same |
| Dlr65-120 / -240 (LR 6.5e-5) | dev32k_1024 / 32k | CE 1.7113 / 1.7102; short 1.1889 / 1.1969 | 1024 | Dlr65->Dlr20 at 120: -0.0260 ±0.0003 | no-cot | cubbins devce/ |
| **Dlr20-120 (LR 2e-5)** | dev32k_1024 / 32k | CE 1.6853 (16-32k 1.6827); short 1.1610 | 1024 | vs stage2 -0.1623 ±0.0044; vs D-120 -0.1332 ±0.0018 | no-cot | cubbins devce/ |
| **Dlr20-240** | dev32k_1024 / 32k | CE 1.6827 (16-32k 1.6797); short 1.1651 | 1024 | vs Dlr20-120 -0.0026 ±0.0001 | no-cot | cubbins devce/ |
| stage3 x8 (OLMo's own long-context model) | dev32k_1024 / 32k | CE 1.8289 (16-32k 1.8014); short 1.1774 | 1024 | — | no-cot | cubbins devce/ |
| stage3 x8 (reference) | devce_dev32k 16k-32k | CE | 256 rows ⚠<500 (4.19M tok) | Δ vs stage2-x4 −0.0455 ±0.0063 paired | no-cot | /data/prasann/olmo3_cptdrop/devce/stage3-yarn8_dev32k.json |
| stage2 x4 (CPT start) | devce_dev32k overall / 0-4k / 4-8k / 8-16k / 16-32k | CE | 256 rows ⚠<500 | — | no-cot | 1.9307 / 1.9698 / 1.9489 / 1.9289 / 1.9173 (stage2-yarn4_dev32k.json) |
| stage3 x8 | same bins | CE | 256 rows ⚠<500 | Δ −0.026±.006 / −0.004±.006 / +0.016±.010 / −0.018±.007 / −0.046±.006 | no-cot | 1.9051 / 1.9661 / 1.9653 / 1.9113 / 1.8718 |
| stage2 no YaRN | same bins | CE | 256 rows ⚠<500 | vs x4: −0.014 / −0.019 at 0-8k, +0.151 / +0.359 at 8-32k | no-cot | 2.1437 / 1.9555 / 1.9297 / 2.0798 / 2.2763 |
| short4k (synthetic math, 1 file) | devce_short4k | CE | 256 rows ⚠<500 | stage3 vs s2x4 +0.0230 ±0.0009 | no-cot | s2 none 0.4643, s2 x4 0.4718, s3 x8 0.4948 |

## Next step
See `next_step` above.

## Open issues / traps
- RESULT 2026-10-24 21:30 LoRA r64 K16 at normal budgets (cubbins 4079261, fresh rows): 240 = 5660 PF (dense 6606) -> 1.6820
  vs 1.6827, overall -0.0007 +-0.0001, far +0.0023, ctx 0.075; 480 = 11296 PF (dense 13188) -> 1.6794 vs 1.6800, overall
  -0.0007 +-0.0001, far +0.0008 +-0.0001, ctx 0.077 (= dense), peak 30.9 GiB; short guard 1.1726 vs 1.1717 (eval_size 256).
  Overall-CE delta vs dense across 12/24/48/120/240/480: -0.0030/-0.0017/-0.0015/-0.0019/-0.0007/-0.0007.
  MoD leftovers (mcfuzz 4024789): c0.35 12/24/48/120 -> 1.7361/1.7266/1.7138/1.7067 (m <=0.47), c0.25-120 1.7143
  (m <0.05), peak 36.4-36.7 GiB (> dense). MoD is out.
- RESULT 2026-10-24 18:20 LoRA r64 K16 + freeze at 75% (cubbins 4063698): 48/120 = 1095/2703 PF (0.81x) -> 1.6868/1.6836
  (unfrozen 1.6864/1.6834), far 0.4818/0.4755 (unfrozen 0.4822/0.4759), ctx 0.072/0.074, peak 30.9 GiB: identical quality
  for 5% fewer FLOPs. Best CPT point: overall -0.0013..-0.0017 vs dense at 0.81x, peak -12%, far +0.0012-0.0013.
- RESULT 2026-10-24 17:30 LoRA r64 (trained sneetches 4030726, scored cubbins 4063697; paired vs dense at same steps):
  K16 (0.849x, peak 30.9 GiB): 12/24/48/120 = 305/587/1151/2842 PF -> 1.6942/1.6895/1.6864/1.6834 (m 1.08/2.27/1.97/1.91);
  overall -0.0030/-0.0017/-0.0015/-0.0019 (+-0.0001); far -0.0026/+0.0013/+0.0015/+0.0012 (+-0.0002); near -0.0019 ..
  -0.0005; ctx 0.068-0.074. K8 (0.924x, 33.1 GiB): 1.6965/1.6910/1.6874/1.6841 (m 0.90/0.96/1.30/1.50); overall
  -0.0007/-0.0002/-0.0005/-0.0013; far -0.0011/+0.0004/+0.0001/-0.0000 (parity). r64 + all-local loses (1.702-1.704,
  ctx 0.043-0.050). Peak saving = frozen params' fp32 grads + Adam state (12 B x 3.6B / 8 GPUs ~5 GiB predicted,
  4.3 measured for K16; activations unchanged under full AC).
- RESULT 2026-10-24 12:00 dense near/far rescore (cubbins 4034568, devce_rep/): far-only CE (4-gram last seen > 4k
  back; 1.25M targets) 12/24/48/120/240/480 = 0.503/0.489/0.481/0.474/0.469/0.467 (2x the overall slope); near
  0.208->0.201; stage2 far 0.715, stage3 0.630. LoRA r16 K16 vs dense, paired: far +0.0066 +-0.0004 (12), +0.0067
  +-0.0002 (48), +0.0040 +-0.0002 (120) = WORSE on long-range targets at every budget (far-only m ~0.6); near -0.0072
  (12) / -0.0005 / -0.0003. The overall-CE win comes from local/novel tokens.
- RESULT 2026-10-24 10:50 all-local training LOSES (cubbins 4021707; --olmo3-train-window 4096; eval full): dense
  12/24/48/120 = 323/621/1217/3005 PF -> 1.7055/1.7037/1.7027/1.7020 (m 0.56/0.33/0.18/0.08); ctx_gain
  0.049/0.047/0.047/0.044 (dense 0.067-0.075, FALLING with training); far CE 0.566-0.576; + freeze K8 f25 same damage
  (m <0.48/0.25/0.17/0.08). The 8 full-attention layers unlearn long-range use within 12 steps.
  LoRA r64: all 16 runs trained on sneetches (4030726), preempted by rhys_gould in the eval stage; exports streamed to
  cubbins for scoring after the dense rescore 4034568. LoRA seed-1 job 4037547 held until then.
- RESULT 2026-10-24 07:30 freeze grid (cubbins 4021612; keep 1.0 warmup-then-freeze of embeddings + blocks [0,K)):
  K16 f25 12/24/48/120 = 278/529/1030/2534 PF -> 1.7094/1.7049/1.6980/1.6881 (m 0.50/0.36/0.27/0.53); K16 f50 ->
  1.7085/1.7010/1.6921/1.6858 (m 0.49/0.42/0.42/1.01); K8 f25 -> 1.7056/1.6992/1.6924/1.6860 (m 0.56/0.44/0.38/0.91);
  K8 f50 -> 1.7064/1.6975/1.6905/1.6851 (m 0.52/0.45/0.63/1.15). Peak 35.9 GiB (pre-freeze); median 26.7 (K16 f25) /
  31.0 (K8 f25). Loses at 12-48 steps, ties only at 120; LoRA r16 K16 beats it at every budget at the same FLOPs.
- RESULT 2026-10-24 06:30 FIRST CPT ARM >= DENSE: LoRA early stack r16 scale 32 (sneetches 4021741; eval re-installs
  adapters; measured PF; dev32k_1024 1,024 rows). K16 (0.843x/step): 12/24/48/120 = 301/579/1135/2804 PF -> CE
  1.6991/1.6921/1.6879/1.6841 (m 0.88/0.82/1.19/1.65), ctx 0.064-0.073, far CE 0.511->0.479, short 1.157-1.159, peak =
  median 30.8 GiB (dense 35.2: -12.5% from step 0). K8 (0.921x): 1.7009/1.6929/1.6880/1.6842 (m 0.75/0.62/1.08/1.45),
  33.0 GiB. Paired: K16-120 vs dense-120 -0.0013 +-0.0001 at 0.85x FLOPs; K16-48 vs dense-48 -0.0000 +-0.0001. A tie or
  small win at less compute AND less peak memory; the large m only reflects the flat dense curve. Run-to-run
  noise: dense seed 1 vs 0 at 120 (different data order: --seed shifts the loader seed 34521+seed; first train losses
  differ) = 0.0000 +-0.0001 on dev CE (ctx_gain 0.0750 vs 0.0743), so the LoRA -0.0013 is ~13x run noise.
  MoD (exports copied mcfuzz -> sneetches): c0.5 12/24/48/120 -> 1.7305/1.7169/1.7050/1.6987 (m <0.45/<0.24/0.16/0.10);
  c0.25 12/24/48 -> 1.7379/1.7337/1.7214: loses. Next: r64 LoRA K16/K8 and r64 + all-local (sneetches 4030726); dense
  near/far rescore (cubbins 4034568).
- TRAP (hit 2026-10-24 01:25): router branch <= 1e5a14c85 never writes config.json["early_lora"] (variants.SoftToken
  defines export_config twice; the later {"softtoken": ...} wins), and DCP loads only the keys the model asks for, so a
  plain-model eval of a LoRA checkpoint silently DROPS the adapters (no strict-load error). DLORA16k16/k8lr20-12 were
  scored that way (1.7086 / 1.7020: INVALID, deleted). Fixed 8d4c861f8: eval_devce rebuilds the block from the lora
  keys + provenance.json and refuses any checkpoint with model keys the eval model lacks (pooled_projector.* allowed:
  every softtoken run carries it, dense eval never uses it). Checked on base / stage3 / dense / OTR / IRAND / SD20 / MoD /
  LoRA checkpoints. Router session told. LoRA relaunched: sneetches 4021741.
- GRAD COMPOSITION ON CPT (grad_composition session, cubbins 4021638; 48 dev rows ⚠ eval_size=48; records/grad-composition.md
  addendum on claude/grad-composition-6f09c7 @36d6c1ce2; artifact https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ):
  blocks 0-6 share of param-grad energy 0.42 base -> 0.20 @12 dense steps -> 0.16 @480 (blocks 0-15: 0.67 -> 0.43-0.46);
  importance FLAT after CPT (top 10% of tokens = 0.62 of G24^2, top 30% = 0.90; top-10% by G7 keeps 0.37, random 0.11,
  pyramid 50/10/3 0.29) -> any token/depth route needs >= 30%; context (K/V) share of a token's hidden-state gradient
  0.95 / 0.80 / 0.60 / 0.42 / 0.21 / 0.12 at blocks 0 / 8 / 12 / 16 / 24 / 28 -> KV-only dropping tolerable from ~block
  20 on; dev-CE gain at >= 4k is 56-66% local 4-gram copying within the last 4k, 6-7% far-only repeats (stage3 vs stage2
  +0.12 on far-only targets at 16-32k): dev CE barely sees long-range loss. Smoke FRZ (K16, freeze after 5 of 20):
  froze 3.65B params at step 6, per-step peak 35.8 -> 26.7 GiB, actual/dense 0.773.
- 2026-10-24: mcfuzz + sneetches staged for this experiment (Prasann via LR-sweep: parallelize): /data/prasann/olmo3_cptdrop/{base/stage2-step47684, data/{train32k_ee,dev32k_1024,short4k_v2}, code_r4780f8a08_o3cd7f406d528, devce_ctx/stage{2,3}*.json}; mcfuzz got the env /data/prasann/conda/envs/corpus-reasoning-olmo copied from sneetches (torch 2.9.1+cu128, flash_attn 2.8.2). Results land in each node's /data/prasann/olmo3_cptdrop/{runs,devce,devce_ctx} -- gather before analysis.
- SCOPING 2026-10-24 (LR-sweep session's KV-only-drop idea: drop tokens only as keys/values in blocks 7-31): from the
  model's FLOP formula (matches the meter: 52.3 vs 52.45 GFLOP/token) attention = 21.0% of step FLOPs at 32k (full
  12.3%, SWA 8.7%), 29.8% at 64k, 42.5% at 128k. Keeping 10% of keys in blocks 7-31 saves 7.4-14.5% at 32k,
  15-22% at 64k, 27-32% at 128k; memory ~0 (full AC, flash). A key-keep flex mask_mod skips no tiles (scattered
  keys); savings need compacted K/V = existing kv_route.py path (full-attn layers only, learned-sigmoid router).
  Recommended: not at 32k; only 64k-128k.
- RESULT 2026-10-24 00:10 exit loss + peak-memory arms (cubbins 4020957, router fcda77090; measured PF; mem = per-step
  peak excluding warmup step 1 (always full, 35.9) / median): exit W1.0 de8 31/62/124/310 = 366/705/1383/3398 PF ->
  1.7366/1.7523/1.7603/1.7706 (WORSE with training; short 1.17->1.24); W0.3 1.7310/1.7354/1.7409/1.7450; de0+exit
  W1.0/0.3 40/80/160 = 386/726/1405 PF -> 1.8476/1.8900/1.9183 / 1.8239/1.8599/1.8816 (worse than the start; mem
  28.6/28.6); F0.5+exit W1 31/62/124 -> 1.7779/1.8059/1.8018; early-nograd+F0.5 46/92/185 = 280/522/1008 PF ->
  1.7433/1.7340/1.7263 (m <0.49/<0.26/<0.14; mem 28.6/22.0, peak -19%); early-nograd de0 60/120/240 = 272/493/935 PF
  -> 1.7804/1.7764/1.7779 (flat; mem 22.0/22.0, -38%). Exit loss bends the SHARED norm + head toward layer-7 states.
  Peak-memory cuts all cost CE; none beats dense. ctx_gain for these pending (CTX stage of 4020957).
- TRAP (hit 2026-10-23 08:57-13:50, cost ~5 h of cubbins): cubbins /data hit 100% (109 trainer stepN checkpoints,
  6.8 TB, ~62 GB each, beside every run's ~65 GB model_and_optim export) and OTR10M7ndx10lr20-160 hung at its final
  save with 0% GPU; the 6 h watcher bound expired before anyone looked. Prasann approved deleting every stepN dir of
  finished runs (exports kept): 5.2 TB free after. driver_lr20 run() now drops stepN of runs without SAVE_INTERVAL
  once exported; the new watcher flags a run log idle > 30 min. Rerun = cubbins 4020957 (OTRX remainder + OTRY).
- RESULT 2026-10-23 07:50 aggressive router cuts (cubbins 4019319; router b24c13423; measured PF; peak/median active GiB):
  early-nograd 46/92/185 = 339/651/1278 PF -> 1.7105/1.7011/1.6975 (m <0.40/0.37/0.22), ctx 0.066/0.069/0.071,
  mem 35.9/22.0 (best router arm); pyramid 3:0.5,7:0.1,16:0.03 40/80/160 -> 1.7430/1.7101/1.7044 (m <0.39/<0.21/0.15),
  36.8/27.9; merge-dropped 31/62/124 -> 1.7315/1.7134/1.7083 (= plain OTR), 35.9/28.6; pyramid+nograd 60/120/240 =
  415/803/1555 PF -> 1.7115/1.7050/1.6989 (m <0.33/0.23/0.17), 36.8/23.7. Peak never drops (every-8 full steps).
  Queued: 4019607 exit-loss (W 1.0/0.3 x de8/de0, + dense-keep 0.5), then 4019842 early-nograd + dense-keep 0.5 and
  early-nograd de0. The F-steps are dense_now() steps, so the early stack DOES train on them (router 15bf04ff3 test); only the de0 arm has a frozen early stack. early-nograd + exit loss dropped: exit grads reach only the head.
- RESULT 2026-10-23 03:40 router CPT (cubbins 4019297; MEASURED flop-meter PF on both sides -- dense is 27.5 PF/step +
  ~27 PF constant, not the 36.6 table value analyze.py used, which flattered drop arms ~1.2x; verdicts unchanged):
  OTR10M7lr20 31/62/124/310 steps = 335/643/1258/3083 PF: CE 1.7308/1.7138/1.7079/1.7055 (m <0.41/<0.21/0.12/0.06),
  ctx_gain 0.0585/0.0672/0.0712/0.0737 (ctx-only m ~0.74-0.79, best of all arms). No-dense-step variant 40/80/160:
  1.8087/1.8010/1.7961 (collapses: upper stack never sees full context). Worse than dense at every position bin
  (0-4k 1.6769 vs Dlr20-48 1.6579): trains on 10% of labels. Memory: routed-step median 28.6 GiB vs dense 35.2
  (-19%), peak 35.9 (set by the every-8 dense steps). Re-measured m (measured PF): IRAND 22/43/65/130/260 <1.25/0.79/
  0.59/0.37/0.20; SD20 0.61/0.38/0.20; IFL25 <0.41/<0.21/<0.10; Dlr10-120 1.95, Dlr05-120 1.66. Next (LR-sweep
  session, for Prasann): early-exit LM loss on dropped tokens at the split layer; smaller micro-batch on dense steps.
- 2026-10-23 router CPT track: online_token_router made marker-free (e0f2ce9ce: block offsets from each 512-block's own
  first/last token; 2 CPU tests, 23/23 pass), merged by the router session into claude/online-router-1bc01a
  (717c4d0ff, +early-nograd / stages / dense-tail / keep-first flags). Smoke OTR 20 steps on 8xH200: actual/dense
  0.407, peak active mem 28.5 GiB/GPU on routed steps vs 35.2 dense. Aggressive-cut job code = router branch
  717c4d0ff + research/token_routing/olmo3_cptdrop from 961e7b7b5 (staged code_r717c4d0ff_o3cd961e7b7b5; branch not
  merged into flopdrop_clean). balrog (A100) NOT used: ask Prasann first.
- FINDING 2026-10-22 08:30 (LR 2e-5, dev32k_1024, 1,024 rows): dense SATURATES almost at once. stage2 1.8477 ->
  Dlr20-12 1.6972 -> 24 1.6912 -> 48 1.6880 -> 120 1.6853 -> 240 1.6827: 12 steps (6.3M tokens) already give
  -0.150 of -0.165, and a 20x compute range (12 -> 240) moves CE 0.015. The gain is uniform across position bins
  (0-4k -0.196, 16-32k -0.144 at 12 steps): adaptation to the pool's text, not long-range learning. Small batch
  (DB8, 8 rows/step, same tokens) 1.6933 / 1.6906 / 1.6901 at 12 / 24 / 48 (helps only at 12). Drop arms at
  matched FLOPs LOSE: IRANDlr20-65 (=12 dense steps) 1.7050, -130 1.7015 (0-4k ties dense, 16-32k +0.011);
  IFL10-133 (~12) 1.7531, IFL10-533 (~48) 1.7701 (worse with more training); IFL05-1040 1.8557 (896-row partial,
  eval hit ENOSPC). C-120 1.6859, C-240 1.6841 (~1.05x). Input dropping damages long-range positions, more so the
  harder it drops. Overall dev CE cannot support a 10-20x reading here, so the eval gained ctx_gain
  (--ctx-window 8192, commit 5408335a6): last 4k targets with full context vs only the last 8k tokens = how much
  the model uses context beyond 8k. CTX stage of cubbins 4017768 scores every run + stage2/stage3 on it.
- 2026-10-22 18:30 last points: Dlr20-480 1.6800 (16-32k 1.6770, ctx_gain 0.0774) -> C-480 1.6810 at 0.945x FLOPs =
  m ~0.8 (C-120 / C-240 ~1.0). 64k (dev64k_512, 512 rows): D64lr20-120 / 240 / 480 = 1.7210 / 1.7170 / 1.7134;
  IFL25x64-600 / -1200 (~114 / ~228 dense-eq, estimated ratio 0.19) = 1.7512 / 1.7507: far above D64-120, m < 1.
  64k dense is a bit steeper than 32k (120 -> 480: -0.0076 vs -0.0053) but still flat.
- VERDICT 2026-10-22 13:00 (analyze.py --ref 'Dlr20|DB8lr20', dev32k_1024, 1,024 rows): NO drop arm reaches
  m > 1 anywhere. Dense: 4 1.7096, 8 1.7008, 12 1.6972, 24 1.6912, 48 1.6880, 120 1.6853 (seed 1: 1.6853), 240
  1.6827. m: IRANDlr20-22 <1.0, -43 0.6, -65 0.5, -130 0.3, -260 0.2; SD20lr20 (sd20 geometry: one detached MEAN
  slot per dropped block; softdetach used cent_cmean, which needs the tokenizer) 65/130/260: 1.7045/1.7011/1.6998
  = m 0.5/0.3/0.2, within 0.0005 of removal; IFLlr20 <=0.3; IFL10 / IFL05 worse with training. ctx_gain ranks
  them the same (SD20-260 0.0659, IRAND-260 0.0648, Dlr20-12 0.0671). C-480 1.6810 (needs Dlr20-480). LR: Dlr10-120
  1.6828 (m 2.0 vs the 2e-5 curve), Dlr05-120 1.6834, Dlr10-12 1.7000 (worse than 2e-5 at 12): the reference
  would be a per-budget LR envelope, which only makes drop arms look worse. Contrast with softdetach (~4x): its
  dense curve fell 0.046 nats/decade of PF; this one ~0.011, so no room below dense.
- FINDING 2026-10-22 10:20 (ctx_gain, --ctx-window 8192, dev32k_1024, 1,024 rows; paired SE): long-context use
  ALSO saturates fast and drop arms trail it. ctx_gain: stage2 0.0169, stage3 0.0456 (OLMo's 50B-token stage:
  +0.0287 ±0.0018), Dlr20-12 0.0671 (+0.0502 ±0.0019 vs stage2), -24 0.0700, -48 0.0726, -120 0.0743, -240 0.0762
  (12 -> 240: +0.0090 ±0.0002, ~+0.002 per doubling). DB8lr20-12 +0.0029 ±0.0001 vs Dlr20-12. Drop arms at
  matched FLOPs: IRANDlr20-65 -0.0054 ±0.0002 vs Dlr20-12; -130 -0.0066 vs Dlr20-24; IRANDlr20-260 (48-step FLOPs)
  still -0.0023 below Dlr20-12; IFLlr20-54 -0.0107. Dlr10-120 (LR 1e-5) overall CE 1.6828 (best at 120 steps)
  with ctx_gain -0.0011 vs Dlr20-120. No m > 1 anywhere yet (both metrics), so no Pareto plot to draw.
- TRAP (hit 2026-10-22 08:33): weka `No space left on device` under prasanns/olmo3_cptdrop: IFL10x64 / IFL05x64 died
  at data staging; IFL10 / IFL05 in-job evals lost chunks. Cause: the WHOLE filesystem (1.7 PB, 97% used, 55 TB
  free at 16:50Z), not a per-user quota. Our olmo3_cptdrop dirs = ~6.4 TB (prasanns/ total 29 TB). Prasann approved
  deleting the superseded old-LR dirs (~2.0 TB: beaker_GEN/C120/C240/C480/IFL, beaker_o3cd-D64/IFL05/IFL10/
  IFL10x64/IFL05x64): Beaker CPU job 01M5JV4926Y7FDYMSE8ABZ1DZ8. Every Beaker job keeps full model+optim checkpoints
  (~88 GB each, max_checkpoints 0) in its own dir, plus superseded old-LR dirs. du job 01M5J458P3MAFN3MPD3CFNKPXQ.
- FIXED in eae16ae0b (branch o3cd/swa-position-window, not merged): compacted rows used to apply the 4,096
  sliding window in compacted index space. SWA layers with gapped position_ids now use a FlexAttention mask by
  ORIGINAL position (built once per forward, ~1 ms at 32k); full-attention layers and contiguous rows stay on
  flash. Keep-all parity <=1e-5 fp32 (bf16 flash bit-identical); leak test; dense-reference outputs + grads;
  361 existing attention/kv_route/soft-token/block-skip tests pass. 32k fwd+bwd: dense unchanged, keep 25%/50%
  ~3% faster. Follow-up: the `block_keep` / kv_route attention path still ignores sliding windows -- fix before
  any OLMo 3 layer-skipping arm.
- RESOLVED (uncommitted trainer change): `olmo3_7B_ctc_swa` took YaRN x8 only; it now takes `yarn_factor`,
  set via `--rope-yarn-factor 4`.
- ~~The eval harnesses silently build Qwen3-4B for an unknown model family~~ FIXED on branch `claude/vigorous-agnesi-20ecb0` (`ctc_devloss_grid.build_cfg` raises); OLMo 3 dev CE = `research/token_routing/olmo3_cptdrop/eval_devce.py` + `compare_devce.py` (paired per-row SE).
- short4k is one preprocessed file (`dolma3_longmino_0625/000500.npy`): LLM-written step-by-step math, CE ~0.47. A narrow forgetting guard; rebuild from several stage-3 files (already in next_step). `ops result` auto-fills a binomial SE (0.031) on those rows: meaningless for CE, use the paired SE.
- Pool length-bucket labels are approximate in the OLMo 3 tokenizer; the builder measures lengths itself.
- Lambda: /data not writable, use node /tmp; sbatch exit codes lie; fair-share 8 preemptive_high + 8 preemptive.
- 7B is not in train_ctc_suite's `_LARGE_SCALES`: pass `--shard-degree 8 --activation-checkpointing full`
  or every GPU holds full fp32 params + Adam state (~117 GB) and OOMs.
- Train topic mix follows the pool (45% science_math_and_technology, 13 of 22 topics sampled); dev is
  topic-balanced over 23 topics. Same for every arm, so arm comparisons are unaffected; absolute CE is not
  comparable to OLMo's own stage-3 numbers.
- TRAP (hit 2026-10-21, cost 80 GPU-min): the trainer splits a shard into documents at EOS and silently DROPS
  any document longer than --seq-len. If no document fits, every epoch is empty and fit() loops forever on
  "Starting epoch N / Epoch complete" with GPUs idle (no error). The first c8k cut 32k rows into 8k pieces with
  only one leading EOS per four, so every document was 32k. Fixed in 5622d24ac (every 8k row = EOS + 8,191).
  Smokes now carry a 20-min timeout.
- TRAP: `--no-final-checkpoint` (run_arm SMOKE=1) disables the whole checkpointer, mid-run saves included.
- TRAP (hit 2026-10-22, every Beaker drop arm): EOS-FIRST rows make the trainer's EOS-split stream start with a
  lone EOS = a degenerate 1-token document (43,001 instances for 43,000 rows). The soft-token variant's
  length-sorted micro-batching gives the rank holding it a different micro-batch structure -> FSDP collectives
  desync -> NCCL watchdog after 15 min. Seed 34521 puts it at step 118 on rank 7 (verified from the dense run's
  global_indices: position 1887 = batch 117, slot 15). IFL / IFL10 / IFL05 all hung at step 118; dense unaffected.
  Fixed 9dbacf5cf: EOS-LAST shards *_ee (eos_to_end.py; same [body, EOS] instances, 43,000 of them) for every
  training shard; Beaker data check requires exactly one EOS per row at the end. Drop arms relaunched (bk6).
- TRAP (hit 2026-10-22): Beaker preempted D64 exactly at its 1h minRuntime and REQUEUED it; the requeue would
  have refused its own step checkpoints (run_arm's anti-auto-resume guard). Fixed 3fc6ae705: beaker_job exports
  RESUME=1 (resume own latest stepN) and skips runs whose export exists. D64 rerun FRESH as D64e (its old
  checkpoints were on EOS-first train64k; resuming onto *_ee data would mix data orders).
- FOR PRASANN: Beaker's preemption notice says our job's workspace group is "ai2/oe-scaling" (15% over target).
  CLAUDE.md says our jobs must never run from an oe-scaling workspace (ai2/flex2 under ai2/oe-other). Launches
  were from -w ai2/flex2 -b ai2/oe-other as documented; worth checking whether flex2 is now in that group.
- FINDING 2026-10-22 03:00: at the OLMo recipe LR (2.07e-4) with our 512k-token batches (OLMo used 4M), dense CPT
  PLATEAUS by ~120 steps (63M tokens) and then gets slightly WORSE (240: +0.010, 480: +0.005 vs 120, paired
  ±0.001), while short-document CE climbs 1.16 -> 1.33 -> 1.39 -> 1.49. Seed noise at 120 steps is 0.009. Most of
  the stage2 -> D-120 gain is the 0-4k bin (-0.078, domain adaptation to PDFs), not long range (16-32k -0.025).
  A flat dense curve cannot support a 10-20x Pareto claim, so: LR probe (6.5e-5, 2e-5; 240 + 120 branch) on
  cubbins (4017622). Beaker jobs at the OLMo LR kept: they sit in the steep 0-120-step regime (DLOW = 24/48).
- TRAP (hit 2026-10-22, ~35 min x 3 Beaker jobs): s3_sync skipped files on equal SIZE, so the rebuilt (same-size)
  c8k never replaced the broken one on S3; C120/C240/C480 livelocked on empty epochs and wrote ~400k
  global_indices_*.npy files each under weka beaker_C*/work/o3cd-C-*-p1/ (+300 MB logs). Fixed 71d519e65
  (mtime metadata / ETag record; beaker_job refuses shards whose rows do not all start with EOS). Weka junk
  cleanup pending until the relaunched C jobs finish phase 1 (same dirs).
- TRAP (hit 2026-10-22): eval_devce refused every TRAINED checkpoint: config.json records the patched OLMo 3 marker
  tokenizer (same ids as dolma2). Fixed df2218269 (accept same-id tokenizers). The cubbins driver 4017567's staged
  copy code_f1cc054c6 was HOT-PATCHED with that eval_devce.py (only file changed) before its EVALD stage. Beaker
  jobs on older commits will fail their in-job eval but still upload exports to S3 -> score centrally.
  DLOW (01M5H9THF152NHBXCYQ5MSAENQ) trained D-48 + D-24 fine (exports on S3 runs/o3cd-DLOW-{24,48}); eval failed.
- Token exit / position rules: GPU tests NOT run (job 4017524 preempted while pending); run before any
  token-exit arm.
- TRAP (hit 2026-10-22): CheckpointerCallback.max_checkpoints defaults to 3, so --save-interval checkpoints older
  than the newest three are deleted. D-480/step108 vanished and the D-120 branch failed; IRAND's saves-every-195
  and six queued Beaker jobs would have lost branch points. Fixed f1cc054c6: train_ctc_suite --max-checkpoints
  (0 = keep all), run_arm passes 0 with SAVE_INTERVAL. D-120 is now a fresh 120-step run (+ seed 1).
- TRAP: WSD decay = round(t_max * decay_fraction); a 0-step decay divides by zero. Only tiny smoke budgets hit it.
- Arm C's two phases use different shards: phase 2 must load model + optimizer only (`--branch-optim-only`),
  never trainer state, or the data loader resumes at a position in the wrong shard.

## Verified vs unverified
- Verified 2026-10-20/21: HF has `stage2-step47684` (8k, no rope scaling) and `stage3-step11921` (65,536,
  YaRN ×8, `sliding_window` 4096 on 3 of 4 layers), 14.6 GB bf16 each; olmo-checkpoints.org serves the
  OLMo-core `model_and_optim` for both. Pool layout: 24 topic folders × 8 length buckets (2^13–2^20), jsonl.zst
  with `id`/`text`; ~490 docs per 2^15 file. Two stage-3 training shards measured: 41% of tokens in docs >8k,
  max doc 132,838 tokens, rows packed to 65,536.
- Verified 2026-10-21: shards (counts, leading EOS, mask, max id, decoded text, train/dev disjoint by doc
  id); stage-2 conversion parity (above).
- UNVERIFIED: everything about training.
