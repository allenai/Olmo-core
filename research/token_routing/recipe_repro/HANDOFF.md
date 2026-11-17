# Hand-off state: recipe reproduction branch (`prasann/recipe-pareto-repro`)

Updated 2026-11-17. Experiment `tokdrop_lr`. Branch cut from `claude/merge-side-branches-1117` @ 8553e80b9.

## What is on this branch beyond the merge branch
- `research/token_routing/recipe_repro/`: README, WHAT_WAS_TRIED, `repro.py` (train / eval / collect / reference /
  check-flags for the mixes fast8k, new4, mt11), `reference.json` + `build_reference.py`, `check_access_beaker.sh`.
- `new4` reference results and collector, checked out from branch `loop/validate` @ 787581912:
  `research/token_routing/early_loss_proxy/validation/{results.json,table.json,collect_val.py,table_val.py,launch_val.py,stage_r4k_rows_beaker.sh,validate_qafter.json}`
  (launch_val.py imports `launch_proxy.py`, which is not on this branch: it is kept for provenance only; `repro.py`
  replaces it). Pool description `research/token_routing/early_loss_proxy/{POOL_V2.md,build_pool_v2.py}` from
  `claude/early-loss-proxy-d6e886` @ d52756c12.
- Eval launcher: r4k rungs for the four new4 subsets (from `loop/validate`); `BEAKER_WORKSPACE` / `BEAKER_BUDGET` env
  overrides. Train launcher: `WANDB_ENTITY`, `WANDB_PROJECT`, `BEAKER_WORKSPACE`, `BEAKER_BUDGET`, `CTC_USER_DIR`
  env overrides (defaults unchanged).
- `validate_suite_ladder_tasks.py`: prefill-length stats (from `loop/validate`).

## Verification
| item | state |
|---|---|
| `repro.py check-flags` (recipe / mt11 recipe / densepf flags == launch_fast8k.py / launch_otr.py) | PASS |
| `repro.py train --dry` for fast8k {recipe,dense}, new4 {recipe,denselora,dense}, mt11 {recipe,dense} | PASS (configs resolve; minRuntime and resume interval as documented) |
| `repro.py eval --dry` for every mix (sample tasks incl. reorder, grouping) | PASS |
| `repro.py reference` for all three mixes (numbers in README section 2) | PASS; cells passed `ops result check` (only warnings: reorder 0.000, 2k cells at 1.000) |
| GPU smoke of recipe + dense on this branch (jsteinhardt slurm) | (not yet run) |
| tiny Beaker launch of the README recipe command | (not yet run) |
| `check_access_beaker.sh` | (not yet run) |

## Jobs
(none yet)
