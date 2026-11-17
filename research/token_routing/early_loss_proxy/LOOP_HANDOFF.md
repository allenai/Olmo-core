# Auto-research loop: state at 2026-11-17 02:00 PT (paused: usage limit reached)

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
