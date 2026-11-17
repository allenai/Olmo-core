---
title: Branch and worktree cleanup inventory, 2026-11-17
date: 2026-11-17
updated: 2026-11-17
experiment: tokdrop_lr
area: token_routing
kind: audit
status: current
superseded_by:
artifact:
summary: Branches and worktrees after building claude/merge-side-branches-1117 (12 side branches merged; only cross-row-alloc and the 10-24..10-30 side branches left out). On 11-17, with Prasann's OK, 15 fully-contained local branches and 2 clean worktrees were deleted (restore file branch-cleanup-deleted-2026-11-17.tsv); 7 remove-now branches were skipped. 17 branches / 15 worktrees become removable once the integration branch is merged.
---
# Branch and worktree cleanup inventory, 2026-11-17

Scope: all local branches of `OLMo-core`, plus the remote `origin/{claude,loop,tmp,prasann}/*` branches
(origin is the shared `allenai/OLMo-core` repo; other users' branches are out of scope). There are no `tmp/*` branches.
Tables reflect the state **after** the 11-17 deletions below.

## Integration branch `claude/merge-side-branches-1117`

Cut from `prasann/flopdrop_clean` @ b6e2b0984; one `--no-ff` merge commit per branch. Not merged into
`prasann/flopdrop_clean` (that needs Prasann).

| merged branch | conflicts and resolution |
|---|---|
| `claude/gpu-workflow-audit-8833c4` | `records/README.md` (regenerated with `ops exp records --write`, as at every merge below) |
| `claude/gallant-robinson-adc54a` (brings `grad-concentration-d91a37`, `grad-routing-gap`) | `experiment_log.json` (kept flopdrop_clean's: 66 entries, a superset of the branch's 61) |
| `claude/cpt-gradmaps-34a768` (brings `grad-support-tokens`) | records index only |
| `claude/mt11-lengen-32k-56ed58` (brings `mt11-lengen-bed337`, `prasann/mt11-64k`) | `ops/experiments/{mt11_64k,learned_router}/STATUS.md` (kept flopdrop_clean's, the newer superset); `fetch_suite_eval_rows.py` (union: `rerank` subset and the `r64k` rung) |
| `claude/online-router-1bc01a` (brings `otr-perrow-floor`) | none |
| `claude/otr-perrow-minfloor`, `claude/posid-reset-0abac3`, `claude/lit-review-efficient-lc-eacf2d` | records index only |
| `claude/oracle-train-bd7414` | `online_token_router.py` and `train_ctc_suite.py`: union of the `score` / `--st-otr-score` choices (`tail` and `oracle`), and both new fp keys (`_r`, `_col`) |
| `claude/docdrop-nongold-42bb6b` (brings `weighted-token-sampling-041da9`) | Preemption resume, per Prasann (11-17): kept the GPU-audit `--resume-interval` / `ResumePointCheckpointer`; dropped `--ephemeral-save-interval` and `CheckpointerCallback.save_final` from 9f173f3e9 (its FLOP-meter / router trainer-state changes kept). `online_token_router.py`: union of the wts sampling path with oracle / tail / `reset_positions`. `merge_tasks.py`: union (`length_mix` pop + `--allow-no-gold`). `tokdrop_lr/STATUS.md`: flopdrop_clean's page plus the docdrop record line and section. `launch_docdrop.py`, `wts/launch_wts.py` now pass `--resume-interval`. |
| `claude/oracle-thresh-sweep-69bc7d` | same resume resolution; `launch_oracle.py` now passes `--resume-interval`; `online_router.py` kept the wts state keys; `build_mt4_fresh_beaker.sh` took the branch's `SKIP_STATS` option |

`--ephemeral-save-interval` -> `--resume-interval` is a direct translation: both keep one rolling ephemeral
checkpoint every N steps and work with `--no-final-checkpoint`; `--resume-interval` additionally skips the step-0
save and deletes the leftover resume point after the `--save-checkpoint` export. Left untouched (history, not
launch paths): the `extra` strings in `docdrop/state.json` and `wts/state.json` (records of past launches), the
comment in `ds64_fast8k/run_eval_lambda.sbatch`, prose in `records/oracle-threshold-sweep.md`, `tokdrop_lr/STATUS.md`,
`make_overview_page.py` and `hub_live/hub.html`.

Still not merged: `claude/cross-row-alloc` (conflicts in `online_token_router.py`, its test file, and
`record_beaker_evals.py`) and the 10-24..10-30 side branches (see table).

Verification: `py_compile` on every changed `.py` file passes; stdlib `ops/tests` 116 tests OK. No GPU or pytest
runs (login node).

## Deleted on 2026-11-17 (Prasann's OK)

Restore file: [`branch-cleanup-deleted-2026-11-17.tsv`](branch-cleanup-deleted-2026-11-17.tsv) (branch, tip SHA,
containing ref; `git branch <name> <sha>` restores). Each tip was re-verified as an ancestor of its containing ref
right before deletion. All were local-only, so no remote branch was deleted.

- `git branch -d`: `claude/agitated-shirley-ef2b23`, `claude/compassionate-pare-542f8e`,
  `claude/confident-kepler-2347e2`, `claude/focused-lovelace-abac0f`, `claude/quirky-booth-51d200`
  (all in flopdrop_clean), `worktree-agent-a2dd42991bb86043e`, `worktree-agent-a7975a8007a7458c7` (in flopdrop_clean).
- `git branch -D` (contained, but not in the deleting checkout's HEAD): `worktree-agent-{a21c5d95be438dc42,
  a480e2836ed465721, a75edca67f4f473ee, ab672bdd08282dc56, abbdf4704a03638b9, ade8bb42e79c67097}` (tip = upstream
  `main` commit ccf4ef4b5, in `origin/main`; no commits of their own), `worktree-agent-aaf0fa578368b3dbd` and
  `worktree-agent-ae84d4c6544508d35` (in `archive/landmark-legacy-2026-10-13`).
- Worktrees removed (`git worktree remove`, no `--force`; clean, only `.claude/settings.local.json` and `__pycache__`
  ignored): `.claude/worktrees/compassionate-pare-542f8e`, `.claude/worktrees/focused-lovelace-abac0f`.
- Skipped from the remove-now list: `agent-attn-explore`, `backup-docchunk`, `landmark-oomfix` (no approved prefix),
  `o3cd/positional-blocks`, `o3cd/swa-position-window`, `o3cd/tokexit-posrules` (`o3cd/` not in the approved set),
  `worktree-agent-acc242851153ec6d7` (checked out in an agent-* worktree). The 5 clean `/tmp` scratch worktrees were
  left by instruction.

## Counts (after the deletions)

| | removable after the integration branch is merged | needs Prasann | remove-now but skipped | keep |
|---|---|---|---|---|
| branches (104) | 17 | 27 | 7 | 53 |
| worktrees (57) | 15 (2 need their results archived first) | 14 | - | 28 (incl. main) |

## Branches

ahead / behind are commit counts against `prasann/flopdrop_clean` @ b6e2b0984. "pushed" compares the local
branch with `origin`.

| branch | last commit | behind / ahead of flopdrop_clean | in integration | pushed | worktree | recommendation |
|---|---|---|---|---|---|---|
| `agent-attn-explore` | 2026-07-08 | 525 / 278 | no | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: prefix outside claude/ worktree-agent/ tmp/ prasann/ |
| `archive/landmark-legacy-2026-10-13` | 2026-10-13 | 525 / 994 | no | no (local only) |  | keep: archive (local only, never pushed) |
| `archive/wt-agent-a2dd42991bb86043e-2026-10-13` | 2026-10-13 | 525 / 276 | no | no (local only) |  | keep: archive (local only, never pushed) |
| `archive/wt-agent-a7975a8007a7458c7-2026-10-13` | 2026-10-13 | 525 / 279 | no | no (local only) |  | keep: archive (local only, never pushed) |
| `archive/wt-agent-aaf0fa578368b3dbd-2026-10-13` | 2026-10-13 | 525 / 278 | no | no (local only) |  | keep: archive (local only, never pushed) |
| `archive/wt-agent-ae84d4c6544508d35-2026-10-13` | 2026-10-13 | 525 / 467 | no | no (local only) |  | keep: archive (local only, never pushed) |
| `backup-docchunk` | 2026-06-26 | 525 / 186 | no | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: prefix outside claude/ worktree-agent/ tmp/ prasann/ |
| `claude/check-viz-branch-cjsmL` | 2026-02-20 | 560 / 1 | no | remote only |  | needs Prasann: remote-only web-session branch off an old main (Jan-Feb 2026); likely delete |
| `claude/cpt-gradmaps-34a768` | 2026-11-10 | 42 / 54 | yes | yes | cpt-gradmaps-34a768 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/cross-row-alloc` | 2026-11-02 | 221 / 328 | no | yes | quirky-booth-51d200 | needs Prasann: merge blocked, online_token_router.py + its test + record_beaker_evals.py conflicts |
| `claude/docdrop-nongold-42bb6b` | 2026-11-13 | 221 / 365 | yes | yes | docdrop-nongold-42bb6b | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/early-loss-proxy-d6e886` | 2026-11-17 | 221 / 420 | no | yes | early-loss-proxy-d6e886 | keep: paused (proxy / auto-research session; Prasann works with it) |
| `claude/fix-todo-mku1qqfx1i7s3cza-gJFpj` | 2026-01-25 | 599 / 1 | no | remote only |  | needs Prasann: remote-only web-session branch off an old main (Jan-Feb 2026); likely delete |
| `claude/gallant-robinson-adc54a` | 2026-11-10 | 1 / 37 | yes | yes | gallant-robinson-adc54a | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/gpu-workflow-audit-8833c4` | 2026-11-10 | 1 / 14 | yes | yes | gpu-workflow-audit-8833c4 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/grad-composition-6f09c7` | 2026-10-24 | 185 / 9 | no | yes | grad-composition-6f09c7 | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive); trial merge: experiment grad_composition still lists it as its branch |
| `claude/grad-concentration-d91a37` | 2026-10-30 | 42 / 12 | yes | yes | grad-concentration-d91a37 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/grad-routing-gap` | 2026-10-30 | 42 / 31 | yes | yes | agent-ab672bdd08282dc56 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/grad-routing-gap-train` | 2026-10-30 | 61 / 347 | no | yes |  | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive); trial merge: trainer + ds64_fast8k state/ledger conflicts |
| `claude/grad-support-tokens` | 2026-11-04 | 42 / 41 | yes | yes | confident-kepler-2347e2 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/great-chatterjee-82e4e0` | 2026-11-17 | 221 / 376 | no | local +2/-0 vs origin | great-chatterjee-82e4e0 | keep: active (hub reorg; local is 2 commits ahead of origin, unpushed) |
| `claude/keepfrac-vs-length-4bf10c` | 2026-11-17 | 221 / 430 | no | yes | keepfrac-vs-length-4bf10c | keep: active (keepfrac-len session running) |
| `claude/kv-only-context-9ee75b` | 2026-10-29 | 57 / 323 | no | yes |  | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/lit-review-efficient-lc-eacf2d` | 2026-11-03 | 5 / 1 | yes | yes | lit-review-efficient-lc-eacf2d | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/merge-side-branches-1117` | 2026-11-17 | 0 / 585 | yes | yes | merge-side-branches-1117 | keep: this integration branch (merge into flopdrop_clean pending Prasann) |
| `claude/mt11-lengen-32k-56ed58` | 2026-11-14 | 221 / 227 | yes | yes | mt11-lengen-32k-56ed58 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/mt11-lengen-bed337` | 2026-11-04 | 221 / 224 | yes | yes | mt11-lengen-bed337 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/multiview-h7cache` | 2026-10-30 | 221 / 335 | no | yes | agent-a21c5d95be438dc42 | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive); trial merge: record_beaker_evals.py conflict |
| `claude/online-router-1bc01a` | 2026-10-28 | 221 / 312 | yes | yes | online-router-1bc01a | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/oracle-thresh-sweep-69bc7d` | 2026-11-11 | 221 / 365 | yes | yes | oracle-thresh-sweep-69bc7d | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/oracle-train-bd7414` | 2026-11-04 | 221 / 349 | yes | yes | oracle-train-bd7414 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/otr-perrow-floor` | 2026-10-28 | 221 / 312 | yes | no (local only) | agent-ade8bb42e79c67097 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/otr-perrow-minfloor` | 2026-11-09 | 221 / 322 | yes | yes | agitated-shirley-ef2b23 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/posid-reset-0abac3` | 2026-11-04 | 221 / 333 | yes | yes | posid-reset-0abac3 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `claude/qwen3-recipe-c876d1` | 2026-10-30 | 61 / 350 | no | yes | qwen3-recipe-c876d1 | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/recipe-cpt-q35-4b` | 2026-11-17 | 221 / 423 | no | yes | recipe-cpt-q35-4b | needs Prasann |
| `claude/recipe-scale-ablation-71d007` | 2026-10-30 | 61 / 331 | no | yes |  | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/run-and-visualize-dQSda` | 2026-02-20 | 560 / 3 | no | remote only |  | needs Prasann: remote-only web-session branch off an old main (Jan-Feb 2026); likely delete |
| `claude/small-model-select-0a941a` | 2026-10-29 | 61 / 1 | no | yes | small-model-select-0a941a | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/sparse-backward` | 2026-10-30 | 61 / 345 | no | yes | agent-a75edca67f4f473ee | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/tokdrop-decay-fix` | 2026-10-30 | 221 / 331 | no | yes | agent-abbdf4704a03638b9 | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive); trial merge: trainer conflict (same resume flag) |
| `claude/train-layerskip-35010b` | 2026-10-28 | 76 / 322 | no | yes | train-layerskip-35010b | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/vibrant-williams-4b0f66` | 2026-10-28 | 76 / 312 | no | yes | vibrant-williams-4b0f66 | needs Prasann: finished 10-24..10-30 side experiment, not merged (merge or archive) |
| `claude/vigorous-agnesi-20ecb0` | 2026-10-21 | 443 / 1 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `claude/weighted-token-sampling-041da9` | 2026-11-11 | 221 / 353 | yes | yes | weighted-token-sampling-041da9 | delete after merge: fully contained in the integration branch (remove its worktree first) |
| `landmark-oomfix` | 2026-07-07 | 525 / 275 | no | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: prefix outside claude/ worktree-agent/ tmp/ prasann/ |
| `loop/alloc` | 2026-11-17 | 221 / 398 | no | yes | agent-a7afab9795fca3d76 | keep: paused loop |
| `loop/beaker` | 2026-11-17 | 221 / 380 | no | yes | agent-a454742f6d0945eb2 | keep: paused loop |
| `loop/cptbase` | 2026-11-17 | 221 / 374 | no | yes | agent-a8e9a5a6830919d27 | keep: paused loop |
| `loop/cptnative` | 2026-11-17 | 221 / 387 | no | yes | agent-aa6fe817f1ff27e7b | keep: paused loop |
| `loop/detach` | 2026-11-17 | 221 / 392 | no | yes | agent-a218e946253e9d5af | keep: paused loop |
| `loop/earlyloss` | 2026-11-17 | 221 / 382 | no | yes | agent-a4dcc32f3526ded8b | keep: paused loop |
| `loop/ffnskip` | 2026-11-17 | 221 / 384 | no | yes | agent-a939eadbd99934460 | keep: paused loop |
| `loop/goldrand` | 2026-11-17 | 221 / 395 | no | yes | agent-a77f5463745e4c3da | keep: paused loop |
| `loop/lambda` | 2026-11-17 | 221 / 377 | no | yes | agent-a4c5eecacb6cf319c | keep: paused loop |
| `loop/lenscale` | 2026-11-17 | 221 / 398 | no | yes | agent-a3ad7c4e953f99c82 | keep: paused loop |
| `loop/longpos` | 2026-11-17 | 221 / 404 | no | yes | agent-a717aab682ac9c264 | keep: paused loop |
| `loop/memscale` | 2026-11-17 | 221 / 401 | no | yes | agent-ac86d3a313c91405e | keep: paused loop |
| `loop/prefix` | 2026-11-17 | 221 / 383 | no | yes | agent-a19c0239f1c9b8a10 | keep: paused loop |
| `loop/rows` | 2026-11-17 | 221 / 398 | no | yes | agent-ac74b7b4cdaa68ee5 | keep: paused loop |
| `loop/validate` | 2026-11-17 | 221 / 395 | no | yes | agent-a36d6a3d399d712e1 | keep: paused loop |
| `o3cd/composite-13f0073d7-offload` | 2026-10-25 | 221 / 245 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/composite-209f4c2d4-offload3` | 2026-10-26 | 221 / 303 | no | no (local only) | wt_bk | needs Prasann: unique commits, local only, unpushed |
| `o3cd/composite-4980d8ecb-e3e05fd97` | 2026-10-25 | 221 / 232 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/composite-51322b89c-0d18060ba` | 2026-10-25 | 221 / 234 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/composite-6f274e339-51ca47b70` | 2026-10-25 | 221 / 233 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/composite-c9f760ffb-offload2` | 2026-10-26 | 221 / 270 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/devce-eval` | 2026-10-21 | 443 / 1 | no | no (local only) |  | needs Prasann: unique commits, local only, unpushed |
| `o3cd/positional-blocks` | 2026-10-21 | 442 / 0 | yes (already in base) | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: o3cd/ prefix not in the approved set |
| `o3cd/swa-position-window` | 2026-10-21 | 442 / 0 | yes (already in base) | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: o3cd/ prefix not in the approved set |
| `o3cd/tokexit-posrules` | 2026-10-22 | 412 / 0 | yes (already in base) | no (local only) |  | remove now (OK'd 11-17) but SKIPPED: o3cd/ prefix not in the approved set |
| `prasann/5task-fixnq` | 2026-07-01 | 525 / 232 | no | yes |  | keep: Prasann's branch |
| `prasann/ctc` | 2026-10-13 | 516 / 103 | no | yes |  | keep: Prasann's branch |
| `prasann/ctc-grader-fixes` | 2026-09-09 | 516 / 96 | no | yes |  | keep: Prasann's branch |
| `prasann/ctc-setA-sft` | 2026-09-23 | 525 / 653 | no | local +0/-1 vs origin |  | keep: Prasann's branch |
| `prasann/ctc-sft-hybridish` | 2026-09-19 | 549 / 65 | no | remote only |  | keep: Prasann's branch |
| `prasann/ctc_public` | 2026-08-20 | 516 / 98 | no | remote only |  | keep: Prasann's branch |
| `prasann/dense32k-base-fix` | 2026-06-26 | 525 / 188 | no | remote only |  | keep: Prasann's branch |
| `prasann/dropstruct` | 2026-10-14 | 471 / 20 | no | no (local only) | .-dropstruct | needs Prasann: Prasann's branch, local only, unpushed |
| `prasann/eval-rerank-lora` | 2026-10-25 | 221 / 244 | no | remote only |  | keep: Prasann's branch |
| `prasann/flopdrop_clean` | 2026-11-17 | 0 / 0 | yes (already in base) | yes | . | keep: base branch |
| `prasann/landmark` | 2026-10-12 | 525 / 993 | no | yes |  | keep: Prasann's branch |
| `prasann/landmark-chunked-prefill` | 2026-10-03 | 525 / 985 | no | yes |  | keep: Prasann's branch |
| `prasann/lengen` | 2026-10-14 | 471 / 14 | no | yes | .-lengen | keep: Prasann's branch |
| `prasann/mt11-64k` | 2026-10-26 | 221 / 202 | yes | yes | mt11-64k | needs Prasann: fully contained in the integration branch (experiment branch; worktree has 379 unarchived result files) |
| `prasann/ops-tools` | 2026-10-12 | 525 / 1 | no | no (local only) |  | needs Prasann: Prasann's branch, local only, unpushed |
| `prasann/sparsity-scale` | 2026-10-21 | 444 / 2 | no | yes | ssc_wt | keep: Prasann's branch |
| `prasann/summtoken-contra-v3fix` | 2026-08-25 | 525 / 626 | no | yes |  | keep: Prasann's branch |
| `worktree-agent-a19c0239f1c9b8a10` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a218e946253e9d5af` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a36d6a3d399d712e1` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a3ad7c4e953f99c82` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a454742f6d0945eb2` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a4c5eecacb6cf319c` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a4dcc32f3526ded8b` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a717aab682ac9c264` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a77f5463745e4c3da` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a7afab9795fca3d76` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a8e9a5a6830919d27` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-a939eadbd99934460` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-aa6fe817f1ff27e7b` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-ac74b7b4cdaa68ee5` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-ac86d3a313c91405e` | 2026-11-14 | 498 / 11 | no | no (local only) |  | keep: paused loop (harness branch of a loop agent; points at an upstream main commit, no unique work) |
| `worktree-agent-acc242851153ec6d7` | 2026-10-22 | 412 / 0 | yes (already in base) | no (local only) | agent-acc242851153ec6d7 | remove now (OK'd 11-17) but SKIPPED: checked out in the agent-acc242851153ec6d7 worktree (agent-* worktrees are left alone) |

## Worktrees

Result files = git-ignored files, excluding `__pycache__`, `*.lock` and `.claude/settings.local.json`.
"copied to archive" checks each of those files (and untracked ones) under
`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core-archive/worktree_files/<worktree>/`. Size is `du -s`
of the checkout (result data lives on node-local `/data` / weka, not here). The main checkout was not scanned.

| worktree | branch | dirty (tracked mods / untracked) | result files (ignored) | copied to archive | size | recommendation |
|---|---|---|---|---|---|---|
| `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core` (main checkout) | prasann/flopdrop_clean | (not scanned: live sessions) | | | | keep |
| `OLMo-core-dropstruct` | prasann/dropstruct | 0 / 0 | 0 | n/a | 14 MB | keep: Prasann's experiment checkout |
| `OLMo-core-lengen` | prasann/lengen | 0 / 0 | 6 | NO | 15 MB | keep: Prasann's experiment checkout |
| `.claude/worktrees/agent-a19c0239f1c9b8a10` | loop/prefix | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a218e946253e9d5af` | loop/detach | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a21c5d95be438dc42` | claude/multiview-h7cache | 0 / 0 | 25 | NO | 16 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: archive results, then remove |
| `.claude/worktrees/agent-a36d6a3d399d712e1` | loop/validate | 0 / 0 | 73 | NO | 19 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a3ad7c4e953f99c82` | loop/lenscale | 0 / 0 | 1 | NO | 18 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a454742f6d0945eb2` | loop/beaker | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a4c5eecacb6cf319c` | loop/lambda | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a4dcc32f3526ded8b` | loop/earlyloss | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a717aab682ac9c264` | loop/longpos | 0 / 0 | 1 | NO | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a75edca67f4f473ee` | claude/sparse-backward | 0 / 0 | 65 | NO | 16 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: archive results, then remove |
| `.claude/worktrees/agent-a77f5463745e4c3da` | loop/goldrand | 0 / 0 | 1 | NO | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a7afab9795fca3d76` | loop/alloc | 0 / 0 | 1 | NO | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a8e9a5a6830919d27` | loop/cptbase | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-a939eadbd99934460` | loop/ffnskip | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-aa6fe817f1ff27e7b` | loop/cptnative | 0 / 0 | 0 | n/a | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-ab672bdd08282dc56` | claude/grad-routing-gap | 0 / 0 | 18 | NO | 27 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: archive results, then remove |
| `.claude/worktrees/agent-abbdf4704a03638b9` | claude/tokdrop-decay-fix | 0 / 0 | 21 | NO | 20 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: archive results, then remove |
| `.claude/worktrees/agent-ac74b7b4cdaa68ee5` | loop/rows | 0 / 0 | 2 | NO | 17 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-ac86d3a313c91405e` | loop/memscale | 0 / 0 | 2 | NO | 18 MB | keep: paused loop (agent worktree, leave alone) |
| `.claude/worktrees/agent-acc242851153ec6d7` | worktree-agent-acc242851153ec6d7 | 0 / 0 | 0 | n/a | 15 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: remove |
| `.claude/worktrees/agent-ade8bb42e79c67097` | claude/otr-perrow-floor | 0 / 0 | 0 | n/a | 15 MB | keep for now (agent-* worktree; not a loop branch). Once the loop closes: remove |
| `.claude/worktrees/agitated-shirley-ef2b23` | claude/otr-perrow-minfloor | 0 / 0 | 42 | NO | 16 MB | remove after merge, but copy its result files to the archive first |
| `.claude/worktrees/confident-kepler-2347e2` | claude/grad-support-tokens | 0 / 0 | 176 | yes 176/176 | 78 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/cpt-gradmaps-34a768` | claude/cpt-gradmaps-34a768 | 0 / 0 | 0 | n/a | 78 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/docdrop-nongold-42bb6b` | claude/docdrop-nongold-42bb6b | 0 / 0 | 0 | n/a | 16 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/early-loss-proxy-d6e886` | claude/early-loss-proxy-d6e886 | 0 / 0 | 2 | NO | 17 MB | keep: paused (proxy / auto-research session; Prasann works with it) |
| `.claude/worktrees/gallant-robinson-adc54a` | claude/gallant-robinson-adc54a | 0 / 0 | 11 | yes 11/11 | 35 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/gpu-workflow-audit-8833c4` | claude/gpu-workflow-audit-8833c4 | 0 / 0 | 25 | yes 25/25 | 23 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/grad-composition-6f09c7` | claude/grad-composition-6f09c7 | 0 / 0 | 0 | dir exists, nothing to copy | 15 MB | needs Prasann |
| `.claude/worktrees/grad-concentration-d91a37` | claude/grad-concentration-d91a37 | 0 / 0 | 23 | NO | 104 MB | remove after merge, but copy its result files to the archive first |
| `.claude/worktrees/great-chatterjee-82e4e0` | claude/great-chatterjee-82e4e0 | 0 / 0 | 0 | n/a | 17 MB | keep: active (hub reorg; local is 2 commits ahead of origin, unpushed) |
| `.claude/worktrees/keepfrac-vs-length-4bf10c` | claude/keepfrac-vs-length-4bf10c | 3 / 1 | 2 | NO | 21 MB | keep: active (keepfrac-len session running) |
| `.claude/worktrees/lit-review-efficient-lc-eacf2d` | claude/lit-review-efficient-lc-eacf2d | 0 / 0 | 0 | n/a | 15 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/merge-side-branches-1117` | claude/merge-side-branches-1117 | 0 / 0 | 0 | n/a | 85 MB | keep until the integration branch is merged |
| `.claude/worktrees/mt11-64k` | prasann/mt11-64k | 0 / 0 | 379 | NO | 22 MB | needs Prasann: archive its 379 result files first; branch is in the integration branch |
| `.claude/worktrees/mt11-lengen-32k-56ed58` | claude/mt11-lengen-32k-56ed58 | 0 / 1 | 145 | yes 146/146 | 17 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/mt11-lengen-bed337` | claude/mt11-lengen-bed337 | 0 / 1 | 74 | yes 75/75 | 16 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/online-router-1bc01a` | claude/online-router-1bc01a | 0 / 0 | 0 | dir exists, nothing to copy | 16 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/oracle-thresh-sweep-69bc7d` | claude/oracle-thresh-sweep-69bc7d | 0 / 0 | 48 | yes 48/48 | 17 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/oracle-train-bd7414` | claude/oracle-train-bd7414 | 0 / 0 | 0 | n/a | 16 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/posid-reset-0abac3` | claude/posid-reset-0abac3 | 0 / 0 | 21 | yes 21/21 | 16 MB | remove after merge: clean, results archived or none |
| `.claude/worktrees/quirky-booth-51d200` | claude/cross-row-alloc | 0 / 0 | 60 | yes 60/60 | 16 MB | keep until the branch is decided; results archived |
| `.claude/worktrees/qwen3-recipe-c876d1` | claude/qwen3-recipe-c876d1 | 2 / 0 | 163 | NO | 22 MB | needs Prasann: dirty; results not archived |
| `.claude/worktrees/recipe-cpt-q35-4b` | claude/recipe-cpt-q35-4b | 0 / 0 | 3 | NO | 21 MB | needs Prasann; results not archived |
| `.claude/worktrees/small-model-select-0a941a` | claude/small-model-select-0a941a | 0 / 0 | 0 | n/a | 15 MB | needs Prasann |
| `.claude/worktrees/train-layerskip-35010b` | claude/train-layerskip-35010b | 1 / 0 | 0 | dir exists, nothing to copy | 16 MB | needs Prasann: dirty |
| `.claude/worktrees/vibrant-williams-4b0f66` | claude/vibrant-williams-4b0f66 | 0 / 0 | 0 | dir exists, nothing to copy | 16 MB | needs Prasann |
| `.claude/worktrees/weighted-token-sampling-041da9` | claude/weighted-token-sampling-041da9 | 0 / 1 | 0 | yes 1/1 | 16 MB | remove after merge: clean, results archived or none |
| `/tmp/.../5812d1ad-720c-4029-b378-14eab0e21279/scratchpad/evalwt` | (detached 67b93b87a) | 0 / 0 | 0 | n/a | 32 MB | needs Prasann: clean scratch worktree in another session's /tmp scratchpad (not removed on 11-17 by instruction) |
| `/tmp/.../5812d1ad-720c-4029-b378-14eab0e21279/scratchpad/evalwt2` | (detached 78ffb1298) | 0 / 0 | 0 | n/a | 32 MB | needs Prasann: clean scratch worktree in another session's /tmp scratchpad (not removed on 11-17 by instruction) |
| `/tmp/.../787b4b22-66f8-4a9c-bf54-527beb8aab30/scratchpad/ssc_wt` | prasann/sparsity-scale | 0 / 0 | 0 | n/a | 32 MB | needs Prasann: clean scratch worktree in another session's /tmp scratchpad (not removed on 11-17 by instruction) |
| `/tmp/.../ca2a8c56-fa2e-4ceb-944a-a08e2d06bfff/scratchpad/wt_bk` | o3cd/composite-209f4c2d4-offload3 | 0 / 0 | 0 | n/a | 33 MB | needs Prasann: clean scratch worktree in another session's /tmp scratchpad (not removed on 11-17 by instruction) |
| `/tmp/.../e0efe4e7-de2f-40f4-a960-8591db6b671b/scratchpad/launch_wt` | (detached 8ce5fd22a) | 26 / 50 | 0 | NO | 33 MB | needs Prasann: dirty scratch worktree of another session |
| `/tmp/.../e0efe4e7-de2f-40f4-a960-8591db6b671b/scratchpad/merge_wt` | (detached 25e0ce6e9) | 0 / 0 | 0 | n/a | 32 MB | needs Prasann: clean scratch worktree in another session's /tmp scratchpad (not removed on 11-17 by instruction) |

## Notes

- `git fetch --prune` during this audit dropped stale remote-tracking refs for other users' deleted remote
  branches (e.g. `origin/vision-donovan-stage-2-training`); no remote branch was touched.
- A new worktree `.claude/worktrees/recipe-pareto-repro` appeared after the scan (another live session); it is not in the table.
- `claude/great-chatterjee-82e4e0` (hub reorg) is 2 commits ahead of its origin copy: unpushed work.
- The `o3cd/*` composite branches, `o3cd/devce-eval`, `claude/vigorous-agnesi-20ecb0`, `prasann/ops-tools` and
  `prasann/dropstruct` hold unique commits that exist only locally.
- Before removing `agitated-shirley-ef2b23` (42 files), `grad-concentration-d91a37` (23) or `mt11-64k` (379), copy
  their ignored result files to the archive like the 11-16 batch.
