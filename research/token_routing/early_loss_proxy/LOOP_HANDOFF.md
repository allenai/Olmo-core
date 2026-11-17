# PAUSED by Prasann at 13:15 on 2026-11-17 ("pause for now"). Heartbeat cron deleted. Do not launch runs or republish the hub tab until he says to resume.
# Tab conflict to fix on resume: the 13:13 republish used a stale copy of the tab builder and reverted layout changes the hub-reorg session had made (it restored them as v113). Use hub_live/build_autoresearch_tab.py from the main checkout directly, never the copy in hub_tab/.

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
validate DONE (14:55, while paused; branch loop/validate @ 787581912 pushed, not merged): generation evals, recipe 3.3x fewer FLOPs than dense LoRA at matched score (8k 3.5x, 16k 3.2x), proxy said 1.9-2.2x; dense full FT f1 0.928 beats recipe f1 0.897 and dense LoRA f1 0.882 (3-task mean, reorder grader broken). hub_tab/validation.json updated, NOT published. On resume: publish, and treat dense full FT as the stronger reference.
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
