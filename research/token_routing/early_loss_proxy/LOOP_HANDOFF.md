# RESUMED by Prasann 2026-11-18 00:20 PT with new rules (was paused 11-17 13:15). Heartbeat cron 13,43 * * * * recreated.
# NEW SCOPE RULE (11-18): LoRA and fixed data re-mixes are part of the DENSE BASELINE. Main methods = dynamic FLOP dropping
# during training only. Direction: start from drop settings that track dense per step at higher step cost, then cut step
# cost gradually. Primary held-out = heldout_v3 (sets gen16k:48,rung16k:32,suite8k:8,suite4k:8; eval_size 384), score with v3.py.
# Hub tab: never use hub_tab/build_autoresearch_tab.py (stale). Use hub_tab/refresh_hub.sh (hub_live builder from the main checkout) after an
# Artifact read of the live page; the hub-reorg session (local_9a6d1692-d798-4df3-b6c6-023470949c95) was unreachable on 11-18.

## IN FLIGHT (wave 1, launched 2026-11-18 00:30-00:45 PT; tag w1; heldout_v3; full epoch; probes 50,100,150,200,300,400,488)
- Beaker (2 GPUs each; `bash fetch_beaker_results.sh w1` then `python3 ladder.py --live --json hub_tab/ladder.json`):
  dense (full FT reference, exp 01M7PW8ST2VPK94RSAK7FTNTN9, ~3.5 h), dense-mix80x3 (dense FT on mtbmix80x3, 1,464 steps,
  01M7PW9JCK4421CR8XR0HA20HQ, up to 10 h), lo-r50to20, lo-c16k50 (LoRA + drop), rc-de4, rc-tail20 (recipe + dense every 4th
  step / dense last 20%), ft-r20to7-de4, ft-r20to7-tail20, ft-c7ramp (cut 7, keep 0.2 -> 0.9), ft-m99 / ft-m95 / ft-m90
  (--st-otr-mass-target, cut 7).
- sneetches slurm 5202808-5202811 (8 GPUs = the whole jsteinhardt preemptive_high cap until they end, ~1-1.5 h):
  ft-r20to7, ft-r50to20, ft-c7k50, ft-c16k50 -> /net/sneetches/data/prasann/early_loss_proxy/results/w1/ (code_16af1bb05).
- 01:25: ft-r20to7 sd0 DONE (0.30x/step, 16k CE 0.138 at step 488), ft-r50to20 sd0 DONE (0.44x, 0.126); dense LoRA is 0.121 at 0.76x.
  ft-c7k50 (0.60x) 0.128 at step 400, ft-c16k50 (0.73x) 0.143 at step 300, still running. Dense FT reference not in yet.
  Freed 4 sneetches GPUs -> slurm 5202886 (dense FT seed 1, ~90 min) and 5202887 (ft-r50to20 sd1 then ft-r20to7 sd1).
  Beaker jobs launched before 01:25 do not stream probe lines (results only when a run ends); later launches do (peek_beaker.sh <tag>).
- 01:40: all four sd0 ladder arms DONE. 16k CE at step 488 / step cost: ft-c16k50 0.118 / 0.72x, ft-c7k50 0.121 / 0.60x,
  ft-r50to20 0.126 / 0.44x, ft-r20to7 0.138 / 0.30x (dense LoRA 0.121 / 0.76x). hub_tab/ladder.json written.
- Agent pool4x (a12e472e83534956c, works in THIS worktree/branch: pull --rebase before pushing): builds mtb16kx4 (31,232 rows,
  1,952 steps) on mooney -> sneetches + weka, for equal-FLOP "more rows" tests of cheap arms against dense FT one epoch.
- 02:00 wave-1 readings (16k CE @488, cost vs dense FT; seed 0): LoRA: lo-c16k50 0.119/0.54x, lo-r50to20 0.125/0.32x, rc-de4 0.129/0.36x,
  rc-tail20 0.131/0.33x, recipe 0.145-0.150/0.22x, dense LoRA 0.121/0.76x. FT: ft-c7ramp 0.113/0.64x (best), ft-c16k50 0.118/0.72x,
  ft-c7k50 0.121/0.60x, ft-r50to20 0.126/0.44x, ft-r20to7-tail20 0.128/0.44x, ft-r20to7-de4 0.130/0.48x, ft-mod8d3 0.137/0.66x, ft-r20to7 0.138/0.30x.
  Still running from w1: dense sd0 (Beaker), dense-mix80x3, ft-m99/m95/m90, ft-mod8s2, sneetches dense sd1 + ft-r50to20 sd1 + ft-r20to7 sd1.
- WAVE 2 (tag w2, Beaker, launched 01:55, streams probes: `bash peek_beaker.sh w2`): lo-r35to12, lo-r50to10, lo-p50-20-7 (7:0.5,16:0.2,24:0.07),
  lo-e50-30-15 (3:0.5,7:0.3,16:0.15), lo-c7ramp (0.2->0.9), lo-c7ramp10to60, ft-c7ramp10to90, seed 1: lo-r50to20, ft-c7ramp, denselora.
- NEXT after dense FT lands: state FLOP ratios vs dense FT at its deepest level; when mtb16kx4 is staged run the tracking rungs
  (lo-r50to20, best ramp) on it at the FLOPs of one dense FT epoch; keep lowering cost from the cheapest rung that tracks.
- 02:35 pool mtb16kx4 DONE (agent pool4x finished, commit ab7640b81; mooney + sneetches /data/prasann/early_loss_proxy/shards/mtb16kx4, weka same rel path).
  EQUAL-FLOP RUNS (Beaker tag x4, --data mtb16kx4, --extra "--max-tokens steps*262144" so the LR schedule spans the arm's own steps, 1,982 PF each):
  lo-r50to20-x4 1,525 steps, lo-c16k50-x4 905, recipe-x4 1,952 (whole pool, no max-tokens), denselora-x4 642 (baseline), ft-r50to20-x4 1,108. ~3-3.5 h.
  CHECK at first probe: LR at step 100 of lo-r50to20-x4 should be ~9.6e-6 (schedule over 1,525 steps); if it is ~9.7e-6+ the token budget maps differently.
  w1 extra readings: ft-m99/m95/m90 0.118/0.122/0.117 at 0.97/0.88/0.79x (no dial: done); ft-mod8s2 0.134/0.77x (mod dropped); ft-r50to20 sd1 0.119.
  w2 at step 150 (live): lo-p50-20-7 +0.007 vs dense LoRA, lo-r50to10 +0.010, lo-r35to12 +0.012, lo-c7ramp +0.014, lo-e50-30-15 +0.061 (early cut at block 3 hurts).
- 02:55 DENSE FT sd0 DONE: 16k CE 0.114 @488 (1,982 PF); curve 50:.255 100:.210 150:.179 200:.161 250:.150 300:.135 350:.129 400:.122 450:.116.
  vs dense FT (ladder.py "dense FT" column): ft-c7ramp 0.113 -> final level x1.59 (ONLY arm at final); step-400 level: ft-r50to20 sd1 x2.04,
  lo-c16k50 x1.74, ft-c7k50 x1.43, dense LoRA x1.20; step-350: lo-r50to20 x2.51; 20/7 budgets only step-250 level (x2.2-2.6).
  Results recorded with ops result add (task heldout_v3_16k, metric dense_answer_ce_step488, 18 rows).
  LAUNCHED 02:58: x4/ft-c7ramp-x4 (762 steps), x4/dense-x4 (976 steps = dense curve to 3,964 PF), w3/ft-c3ramp, w3/ft-c7ramp10to60, w3/ft-c7ramp-tail10.
  Agent keepsched (adb93081e36fd6262, branch loop/keepsched, Beaker tag ks): piecewise keep schedule for ALL stages; hold-then-ramp shapes at <=0.45x.
  x4 schedule check passed (lr 9.63e-6 at step 100 of 1,525).
- 04:40 x4 mid-run (python3 eqflop.py --json hub_tab/eqflop.json = equal-PF table; live PF = steps x COST table in eqflop.py): cheap arms far below dense at
  equal PF (lo-r50to20-x4 0.091 @1,250 PF vs dense 0.135-0.142; recipe-x4 passes dense final ~800 PF). Launched x4/dense-x4full + x4/denselora-x4full
  (4 GPUs, 1,952 steps, ~5 h) for the dense curve to 7,930 PF; x4/dense-x4 (976 steps) ends ~08:00. KEY NEXT: when x4 arms finish (05:30-07:30) and dense
  curves extend, state FLOP ratio at the deepest dense loss; then push cheaper arms on even more rows (need a bigger pool: x8/x16 via build_pool_v2.py
  --gen-rows; qdmatch_nq clean rows are the limit: ~3,100 of 8,000 generated per rung).
  One-epoch: w3 ft-c7ramp-tail10 0.115/0.65x, ft-c7ramp10to60 0.125/0.48x, ft-c3ramp 0.132/0.59x, ft-r50to20-ex50 0.123, ft-r50to20-rnd 0.132 (router matters);
  ks: hold50 0.120/0.46x, hold70 0.124/0.41x, r2ramp 0.120/0.61x, rlow 0.122/0.44x; gc at 20/7: worse than none (invmass 0.153). dense LoRA sd1 0.118.
- 03:35 W2 DONE. dense FT sd1 0.119 (sd0 0.114). ft-c7ramp sd1 0.119 (= dense sd1): ramp x1.57-1.59 at dense final on BOTH seeds. ft-c7ramp10to90 0.119/0.60x.
  LoRA: lo-c7ramp 0.120/0.48x, lo-c7ramp10to60 0.124/0.36x, lo-p50-20-7 0.123/0.29x, lo-r50to10 0.123/0.28x, lo-r50to20 sd1 0.123/0.32x, lo-r35to12 0.131/0.27x,
  lo-e50-30-15 0.152/0.23x (cut at block 3 hurts; w3/ft-c3ramp also far behind). ft-r20to7 sd1 0.135. All recorded with ops result add.
  Launched 03:30: w3/lo-p50-20-7 sd1, w3/lo-p50-10-5, w3/lo-q50-30-15-7 (7,12,16,24), x4/lo-p50-20-7-x4 (1,683 steps).
  Cost floor note: blocks 0-6 run on all tokens = 0.22x (FT) / 0.16x (LoRA) of dense; going below ~0.25x needs a cheap way through the early blocks
  (router cut at block 3 fails). Candidate next ideas: early blocks at reduced keep chosen by a block-7 router from the previous step / cached scores;
  early-block gradient only for tokens kept at the first cut.
- 03:20 launched w3/ft-r50to20-ex50 (explore 0.5), w3/ft-r50to20-rnd (--st-otr-score random). Agent gradcomp (ae04a33f56958de1a, branch loop/gradcomp,
  Beaker tag gc): backward-only gradient rescale of kept document tokens at each cut (inv_keep / inv_mass / sqrt). Agent keepsched still running (tag ks).
  IN FLIGHT SUMMARY: w1 dense sd1 + dense-mix80x3 + ft-r20to7 sd1; w2 (10 arms, ends ~03:40-04:30); w3 (5 arms); x4 (7 arms incl. dense-x4, ends ~06:00-07:30);
  ks (agent); gc (agent). ladder.py reads v3, w*, x4, sbw, ks; add gc to SRC when its runs exist.
- sparsebwd DONE 03:10 (branch loop/sparsebwd @ d5457c575, flag --st-otr-sparse-bwd, not merged): dense forward + gradient only through kept tokens
  = 0.136 @488 at 0.54x (fwd-drop same budgets 0.138 at 0.30x; dense 0.114). DIAGNOSIS: the gap is the missing GRADIENT of dropped tokens, not the
  truncated context. So: number of gradient-carrying tokens (and when) is the lever -> keep ramps / schedules; fwd dropping is the efficient form.
- ft-* arms = softtoken, LR 5e-6, RECIPE without the three --st-otr-early-lora* flags (full fine-tune + drop, 0.30x dense/step).
- Scoring: ladder.py (same-adapter dense baseline: ft-* -> w1/dense, LoRA arms -> v3/denselora, *-mix80x3 -> w1/dense-mix80x3).
  Known from v3: recipe (LoRA) is +0.025-0.03 CE behind dense LoRA at 16k at every probe 150-488.
- NEXT: read which rungs track dense at steps 300-488; take the cheapest that does and lower its cost in small moves
  (more arms between it and the first rung that lags); FLOP-matched 203 PF screens (--max-pf 203 --pf-points 50,100,150
  --steps 2000, smoke passed) for new mechanisms; second seed for any arm within 0.01 CE of dense at step 488.
- Hub: last published from this session as version 123 at 03:40 (first 121 at 02:15) (hub-reorg session unreachable). Procedure: Artifact read of the hub URL, `bash hub_tab/refresh_hub.sh <saved page>`, Artifact publish of the printed new.html to the same URL (no force). The ladder table comes from hub_tab/ladder.json via hub_tab/add_ladder_table.py (interim; should move into the hub builder; new idea ids have no implementation notes in make_idea_impl.py yet).

# Auto-research loop: state file (updated 2026-11-17 12:05 PT)

PRIORITY (Prasann 11-17 12:00): 16k held-out loss first; ideas that gain more as rows get longer. Heartbeat cron 13,43 * * * * restarts the loop.
Best at 12:00: recipe on mtbmix80 3.1-3.4x vs dense LoRA at CE 0.125 (all lengths), but only ~1.2x over the recipe at 16k; recipe 2.1-2.7x at 16k.
In flight at 12:05: Beaker deep2 (full-epoch rc-mix80 x2, rc-mix50), sneetches m80 (k10, k30, gb32, lr2e5), agents alloc (a7afab9795fca3d76), lenscale (a3ad7c4e953f99c82), data (a4c7431e958c3b6ce: mix90/95/80L/80h pools then heldout_v3), validate (a36d6a3d399d712e1: evals running, table in 3-5 h), hub tab (a8a8bae49778d6cbb).
heldout_v3 EXISTS (mooney, sneetches, weka, Lambda). Beaker tag `v3` (launched 11:57): denselora, denselora2e5, recipe sd0/sd1, rc-mix80 sd0/sd1, rc-mix50, rc-mix80L, rc-mix90, rc-mix95, rc-mix80h, all with --sets gen16k:48,rung16k:32,suite8k:8,suite4k:8, probes 50..488. TODO when they finish (~13:00-14:00; denselora ~14:00): fetch tag v3, extend depth_set.py to take gen16k+rung16k as the primary key and v3/denselora as the reference, write the 16k table to the tracker. New pools: mtbmix90, mtbmix95, mtbmix80L (long part all 16k), mtbmix80h (short part all 1k). m80 variants on sneetches (k10, k30, gb32, lr2e5): standard keep and k30 best; others behind.
MEMORY GOAL (Prasann 11-17 12:15): flat memory at 256k-1M+, not savings at 16k; agent memscale (ac86d3a313c91405e).
lenscale: dlA (block 3 local) ~recipe -0.01..0.02 CE/step; dlB/dlC stall; control smoke Beaker 01M7NJQFFRJ967Q32TNDRK0DZM (tag smoke-lenscale2: dlBstd vs dlB, read [doclocal-check] lines in its log). alloc: closed, no gain at 16k. memscale: scaling law measured; transient_gib in dense_probe was wrong (fix in loop/memscale, merge it).
deep2 full-epoch: rc-mix80 3.1-3.7x all lengths, 3.2-3.5x on suite16k (56 rows); rc-mix50 2.4-2.6x. Data agent building mtbmix80x3 (5,856 rows/task) -> run recipe on it full epoch (1,464 steps, Beaker, heldout_v3) when on weka.
lenscale CLOSED negative (doc-local attention hurts; block 3 only is mild but worse per FLOP at 16k). New agent longpos (a717aab682ac9c264): position stretching on short rows + position-preserving doc subsampling of long rows. v3 tag also has rc-mix80x3 sd0/sd1 (1,464 steps).
HUB REORG DONE (v108, 13:00): hub has an Overview + six groups; auto-research tab is in group 3. New builder copied to hub_tab/build_autoresearch_tab.py (markers AR:PART / AR:SCRIPT; source of truth hub_live/build_autoresearch_tab.py in the main checkout). Refresh: python3 hub_tab/build_autoresearch_tab.py HUB.html --state hub_tab/loop_state.json; node --check; publish. Republishes resumed via the hub-tab agent.
memscale DONE and merged (flags --prefix-chunk N, --prefix-chunk-offload, --st-otr-keep-cap N; dense_probe memory fix): slope 0.040 GiB/1k with chunk 4k + cap 4096 + offload; 1M row fits (77.5 GiB), 2M at 126.5 GiB; 16k learning = recipe. NEXT for memory: learning test at 64k+ rows with chunking + keep cap (needs a long-row pool with real tasks), keep-cap effect on learning at 16k.
longpos DONE (13:40, while paused): doc subsampling with preserved positions keep 25% = 1.23-1.35x over recipe at 16k (ties rc-mix80); stretching no help. NEXT on resume: subsample the long rows of mtbmix80 (keep 0.25 / 0.1), flags --longpos-docsub-keep in branch loop/longpos (not merged).
validate DONE (13:58, while paused; branch loop/validate @ 787581912 pushed, not merged): generation evals, recipe 3.3x fewer FLOPs than dense LoRA at matched score (8k 3.5x, 16k 3.2x), proxy said 1.9-2.2x; dense full FT f1 0.928 beats recipe f1 0.897 and dense LoRA f1 0.882 (3-task mean, reorder grader broken). hub_tab/validation.json updated, NOT published. On resume: publish, and treat dense full FT as the stronger reference.
SCORING RULE CHANGED (Prasann 15:00): only the deepest dense loss counts (methods must scale to large data; extra steps fine). deep.py -> hub_tab/deep.json: 0 of 112 runs reach dense LoRA's full-epoch loss (0.071 at 1,496 PF); deepest = plain recipe full epoch, dense step-250 level (0.098) at 2.2-2.6x. v3 (16k, 320 rows) fetched and scored: dense LoRA full epoch 0.121 at 1,496 PF; recipe full epoch 0.150 / 0.145 at 442 PF; rc-mix80x3 (1,464 steps) 0.136 / 0.134 at 479 PF; both reach only dense's step-300 level (0.146) at 2.1-2.4x. Short-row mix advantage at 16k is gone at depth (mix80x3 ~ recipe at equal PF). On resume: long runs sized to reach the dense final loss (recipe / mix80 with 2-3x rows), not more 100-step screens; hub bars must use the deepest-level ratio.
SCREEN BUDGET (Prasann 15:20): FLOP-matched screens, budget = 50 dense steps = 203 PF; cheaper arms get proportionally more steps. New flags --max-pf / --pf-points in dense_probe + all three launchers (UNTESTED: smoke first on resume). Existing runs at 203 PF, all lengths: dense FT 0.177, dense LoRA 0.151-0.166, recipe 0.114-0.127, rc-mix50 0.105-0.110; 16k sets: dense LoRA 0.24, recipe 0.195-0.197, rc-mix50 0.181, rc-mix80L 0.189, rc-mix80x3 0.192-0.200. Most screening arms stopped far below 203 PF and are unjudged under this rule.
REFERENCE = DENSE FULL FT, not dense LoRA (Prasann 15:40). deep.py --ref v2/dense (hub_tab/deep_densefull.json): dense FT curve exists only to step 100 (0.135 at 409 PF); recipe 2.4-2.8x there, mix50 3.3-3.6x. FIRST LAUNCH ON RESUME: full-epoch dense full-FT proxy run with probes on heldout_v2 and heldout_v3 (4 GPUs Beaker, ~1 h, 1,982 PF), then rescore everything against it; also recipe / mix80 on 2-3x rows to see whether they reach its final loss.
Tools: rescore.sh, iso.py, iso_set.py set/suite16k, depth.py, depth_set.py suite16k.

(older state below)


Resume: `./rescore.sh` (BK_TAGS="deep v2s1 mq cptnative" for extra Beaker tags), `python3 iso.py | head -60`, update
`hub_tab/{ideas,hourly,notes,arms,validation}.json`, `python3 make_loop_state.py --boards hub_tab/board_v2.json --pilot hub_tab/board_s50.json`,
then republish hub tab 8d (builder `hub_tab/build_autoresearch_tab.py`, hub https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt).

## Result so far (one seed, eval_size 184 held-out rows, below 500)
- Reference = cheaper of dense LoRA LR 1e-5 / 2e-5: held-out CE 0.170 @50, 0.126 @100, 0.74x dense per step.
- Recipe: 0.22x dense/step, 1.7x fewer FLOPs to the 100-step level (1.4x worst task). Still the best single method.
- Ahead of the recipe at matched loss: recipe on mtbmix50 1.25x, mtbmix80 1.24x, gold-aware keep grJ 1.26x,
  rows <= 5k 1.35x (shallow, 66 steps). Nothing near 10x.
- Negative: no-grad prefix / earlier cuts, FFN skip, early-exit-loss selection, detached context, input-level token
  exit (std and drop-CPT base), token exit at block 8 from the CPT base (tie), LR 2e-5 / 4e-5, 8 rows/step,
  keep below 20%/7%, random keep, short rows alone.

## Not yet scored / in flight when paused
- Beaker `deep` (rc-mix50, rc-mix80, seeds 0/1), `v2s1` (seed-1 recipe + dense LoRA 2e-5), `mq` (recipe / dense LoRA
  on multi-question pools mtb16k_mq, mtbmix50_mq), `full` (488-step dense LoRA, recipe x2), `short/rc-mix50-lr2e5`,
  `detach` memory arms (acn_*), `cptbase` on mooney (router recipe from L8fl / dense-CPT bases).
- Agents (worktrees under .claude/worktrees/agent-*): goldrand a77f5463745e4c3da (grJ deeper + mixed pools),
  rows ac74b7b4cdaa68ee5 (token-budget batches, length curriculum), alloc a7afab9795fca3d76 (loss-proportional
  sampling), validate a36d6a3d399d712e1 (7 full training jobs on Beaker + generation evals; branch loop/validate).
- Validation Beaker jobs: dense LoRA f1/f2/f4, recipe f1 x2 + f2, dense FT f1 (ids in the agent's interim report /
  `ops status`); evals still to launch if the agent was cut off.
