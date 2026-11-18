# Hand-off state: recipe reproduction docs

Users start from `prasann/flopdrop_clean` (fast-forwarded to this work) and cut their own branch. The work branch for
these docs is `prasann/recipe-pareto-repro`; the coordinator fast-forwards `prasann/flopdrop_clean` after each push.

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
| `check_access_beaker.sh` | PASS: Beaker 01M7NZ6PJ6W5PFRXXTGT0SGB98, every path OK (jobs run as root; base files mode 600, data world-readable) |

## Jobs
| what | id | state |
|---|---|---|
| GPU smoke, recipe + dense, 8 steps each, new4 geometry (smoke_slurm.sbatch, mooney, code /data/prasann/recipe_repro/code_3745da2af) | slurm 5202048 | pending (mooney full); log /data/prasann/joblogs/rr-smoke-5202048.out |
| README recipe command, fast8k f488 (122 steps) | Beaker 01M7NZ7K87C25SQ91TRYFFFFCW (rr-fast8k-recipe-f488-bksmoke) | DONE exit 0: 15 min on 2 H100, 86.5 PF (= 345/4, as expected for 1/4 of f1952), keep_frac 0.205, s1_keep_frac_abs 0.078-0.080, peak 26.9-27.1 GiB, train CE 1.22 -> ~0.15-0.25; model-only export written |
| evals of it (4 tasks x 8k/2k) | Beaker 01M7P0DCR9J8ZRC00W6X90DFP7 / 01M7P0EBRBX4SJ1E7CBX8FX9VP / 01M7P0F5WBZWPN507MT2BKPCH9 / 01M7P0FZP9TGME2T9PE3F26RA0 | launched |
| dense fast8k f488 (comparable to reference dense f488 0.667 8k) | Beaker 01M7NZBWV3PFC2HH52W7GWKGFD (rr-fast8k-dense-f488-bksmoke) | pending |

Next when they finish: `repro.py eval --mix fast8k --run <run>`, then `repro.py collect --mix fast8k`; compare dense
f488 with the reference row (0.667 at 379 PF).
