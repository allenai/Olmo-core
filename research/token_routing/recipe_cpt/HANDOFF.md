# recipe_cpt handoff (current state; keep this file current)

Updated 2026-11-17 16:25 PT. Branch `claude/recipe-cpt-q35-4b` (worktree `.claude/worktrees/recipe-cpt-q35-4b`), cut
from `origin/claude/early-loss-proxy-d6e886` @ d52756c12. Design and smoke numbers: `README.md`.

## Running (Beaker, ai2/flex2, budget ai2/oe-other, urgent, jupiter, 1 node x 8 H100)

| run | Beaker id | expected | minRuntime | save folder |
|---|---|---|---|---|
| `rcpt-q35-4b-recipe-u1B` | 01M7P0EPDSS7CSXED8MCCJNGEX | ~114 min (3.30 s/step x 1,938 + startup/saves) | 2h30m | `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/rcpt-q35-4b-recipe-u1B` |
| `rcpt-q35-4b-dense-u1B` (control) | 01M7P00RGZSA3GF23WDY5EVD8K | ~279 min (8.38 s/step) | 6h05m (Beaker accepted >4h: a 7h test job scheduled in 2 min) | `.../ctc_suite/ckpts/rcpt-q35-4b-dense-u1B` |

wandb group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/rcpt-q35-4b

Checkpoints (`<save folder>/step<N>/model_and_optim`; tokens = N x 16 x 32,257):

| step | tokens | role |
|---|---|---|
| 242 | 124.9M | base "125M" |
| 484 | 249.8M | base "250M" |
| 968 | 499.6M | base "500M" |
| 1936 | 999.2M | base "1B" (the post-fit export `<save folder>/model_and_optim` = step 1938, 1.000B, after the WSD decay; prefer it) |
| 726, 1210, 1452, 1694 | | resume points only: delete after the run (created by this task) |

Ephemeral resume checkpoints every 121 steps (newest only). A preemption restart resumes on its own; a manual
relaunch: `python research/token_routing/recipe_cpt/launch_recipe_cpt.py --arms <arm> --resume --min-runtime <..> launch`.

## Next (in order)

1. As each checkpoint lands (check with `ops status`, or the save folder via a Beaker job), export it and score
   dev CE, one 1-GPU job each (~6 min). Recipe:
   `RUN=rcpt-q35-4b-recipe-u1B STEP=242 ARM=recipe bash research/token_routing/recipe_cpt/post_ckpt_beaker.sh`.
   Dense: `ARM=dense`. Final: `STEP=final`. Gantry must be on PATH:
   `export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH`.
   Register each with `ops register --kind eval`; record with
   `ops result add --experiment recipe_cpt --task cpt_dev --rung 64k --metric full_ce --eval-size 32 --cot-label no-cot --check`.
   Output: the export lands in `/weka/.../prasanns/recipe_cpt/bases/<run>-step<N>/model_and_optim`; the JSON in
   `/weka/.../prasanns/recipe_cpt/devloss/<run>-step<N>.json`.
2. Downstream SFT from an exported base (8k primary screen, mt4 s1952, flags identical to the reference runs):
   `python research/token_routing/recipe_cpt/launch_sft_from_cpt.py --base-tag rcpt-q35-4b-recipe-u1B-step242 --arms recipe,dense launch`,
   then the same command with `eval` once the runs finish. Dense-from-same-base points for an FTM curve:
   `--dense-budgets s976,s3904`. The same command works for the dense-CPT bases (`--base-tag rcpt-q35-4b-dense-u1B-step242`).
3. Delete the extra resume checkpoints and the smoke/probe folders after the runs (all created by this task):
   `ctc_suite/ckpts/rcpt-q35-4b-{recipe,dense}-u1B-smoke{,2}`, `ctc_suite/ckpts/rcpt-q35-4b-*-probe-*`,
   `recipe_cpt/bases/*-smoke-stepfinal`, `recipe_cpt/bases/*-probe-*`.

## Done

- Shard `recipe_cpt/shards/cpt32k_u1B` built and checked (31,008 rows, dev part 28 disjoint), job 01M7NXK410A5T1MVFKK4RKPZ2P.
- Smokes: recipe/dense `-smoke` (no mid-run checkpoint: trap, see README), `-smoke2` A (20 steps, saves at 5/10/15/20)
  and B (`--resume`, 20 -> 30). Merge check, base dev CE, LR probes and loradense probe: README "Smoke and probes".
- Dev CE recorded in `ops result` (experiment recipe_cpt): base 1.3199 ⚠ eval_size=32. Smoke and probes ⚠
  eval_size=8.
- No dense-CPT control from softdetach survives: `sdcpt-q35-4b-dense-u{4,8,16,32,64,128}M` were deleted in the
  2026-10-30 weka cleanup (tier A, `ops/experiments/weka_cleanup/manifests_20261030/APPROVED_A_B_C.tsv`). They were
  64k rows, LR 3e-5, 8 rows/step anyway, so they did not match this setup. The control runs fresh.

## Open issues

- Recipe CPT raises full-context dev CE by +0.12 to +0.17 within 20 steps at every LR tried (README). If the 125M
  checkpoint shows the same, decide before SFT-ing from it. Options (not launched):
  - recipe + periodic dense steps (`--st-otr-dense-every N`);
  - an early-exit loss on dropped tokens;
  - loradense CPT as the "LoRA without routing" base.
- The intermediate checkpoints sit at the constant WSD LR (not annealed). Only 1B is annealed.
