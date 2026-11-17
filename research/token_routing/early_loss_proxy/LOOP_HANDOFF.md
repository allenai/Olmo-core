# Auto-research loop: state file (updated 2026-11-17 12:05 PT)

PRIORITY (Prasann 11-17 12:00): 16k held-out loss first; ideas that gain more as rows get longer. Heartbeat cron 13,43 * * * * restarts the loop.
Best at 12:00: recipe on mtbmix80 3.1-3.4x vs dense LoRA at CE 0.125 (all lengths), but only ~1.2x over the recipe at 16k; recipe 2.1-2.7x at 16k.
In flight at 12:05: Beaker deep2 (full-epoch rc-mix80 x2, rc-mix50), sneetches m80 (k10, k30, gb32, lr2e5), agents alloc (a7afab9795fca3d76), lenscale (a3ad7c4e953f99c82), data (a4c7431e958c3b6ce: mix90/95/80L/80h pools then heldout_v3), validate (a36d6a3d399d712e1: evals running, table in 3-5 h), hub tab (a8a8bae49778d6cbb).
TODO when heldout_v3 exists: rerun dense LoRA + recipe full-epoch references with --heldout heldout_v3 --sets gen16k:48,rung16k:32,suite8k:8,suite4k:8; make 16k the primary key in score.py / depth.py.
MEMORY GOAL (Prasann 11-17 12:15): flat memory at 256k-1M+, not savings at 16k; agent memscale (ac86d3a313c91405e).
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
