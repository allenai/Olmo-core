---
title: Handoff — FLOP-drop main agent (tokdrop_lr and its side threads), state at 2026-11-17 14:45 PT
date: 2026-11-17
updated: 2026-11-17
experiment: tokdrop_lr
area: token_routing
kind: handoff
status: current
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Start here to take over as the FLOP-drop main agent. Three side sessions are open (keep fraction vs context length 2k-64k, running; the early-loss proxy / auto-research loop, paused by Prasann; the hub reorganization, idle), two Beaker jobs and four queued slurm jobs belong to the paused loop, and several decisions wait on Prasann. Lists every active thread with its session, branch, jobs, what is checked vs only reported, what to do when each reports, and the rules and traps from 11-10 to 11-17. Supersedes tokdrop-handoff-2026-11-09.md for current state; that doc still holds the goal, recipe and mechanism background.
---
# Handoff: FLOP-drop main agent (2026-11-17)

You are taking over as the **main orchestrating agent** for Prasann's FLOP-drop work (experiment
`tokdrop_lr`, repo `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core`, branch
`prasann/flopdrop_clean`). The main agent plans, hands experiments to Opus 5.5 side sessions or subagents,
checks their numbers before relaying them, and keeps the experiment log, the hub and STATUS.md current.

Read in this order: this file, then `records/tokdrop-handoff-2026-11-09.md` (goal, recipe, mechanism picture,
what was ruled out before 11-09: still valid), then `CLAUDE.md` and the memory index. Section 7 lists what
changed in the rules since 11-09.

Provenance marks used below: **[checked]** = I compared the numbers against `ops` or the branch's result
files; **[reported]** = a side session's claim that I did not check.

## 1. Goal and standing preferences (Prasann)

- 10x training-FLOP speedup and the lowest peak memory for long-context fine-tuning of Qwen3.5-4B, dense
  inference. Nothing is near 10x yet.
- Every idea gets two numbers: best-case per-step FLOP speedup, and empirical FLOPs to match dense. A
  mismatch is fine if the method is still Pareto-optimal.
- No methods that rely on dense training steps (peak memory counts as much as FLOPs).
- Soft preference for a principle or end-to-end algorithm over hand-set knobs; the two-stage recipe "still
  seems a bit hacky".
- The main agent orchestrates and analyzes; Opus 5.5 side sessions / subagents do most of the work.
- 11-17: priority is the 16k (and longer) held-out loss and methods that gain more as rows get longer;
  memory goal is flat memory at 256k to 1M+ tokens, not savings at 16k.

## 2. Threads that are open right now

### 2a. Keep fraction vs context length (RUNNING, started 11-17 14:20)

- Session: "Keep fraction vs context length (32k/64k proxy)", id `local_80f831eb-037a-44c2-a275-9341a27e99e8`,
  name `keepfrac-len`, Opus 5.5. Worktree `.claude/worktrees/keepfrac-vs-length-4bf10c`, branch
  `claude/keepfrac-vs-length-4bf10c` (cut from `claude/early-loss-proxy-d6e886`).
- Asked by Prasann: does the recipe's best keep fraction get lower as context grows (32k and 64k first), on
  the cheap dense-held-out-loss metric; plus the gold-aware stochastic keep idea from auto-research. A
  second message added 2k, 4k, 8k and 16k under the same protocol (it was queued behind the session's turn;
  confirm it was picked up).
- Design in its brief: single-length training pool and same-length held-out set per length; arms dense-LoRA
  reference, 20%/7% (recipe), 40%/14%, 10%/3.5%, 5%/1.75%, 2.5%/0.9% (1.25%/0.45% at 64k if still
  improving; a higher-keep arm at short lengths if 40%/14% wins); two seeds on the deciding arms;
  gold-aware stochastic keep `grJ` plus one outlier-safe variant at 32k and 64k.
- Deliverables: `records/keep-fraction-vs-length.md`, a chart (FLOP speedup at matched loss vs keep
  fraction, one line per length, plus best keep vs length), `ops result add --check`,
  `research/token_routing/early_loss_proxy/keepfrac_len/experiment_log_entry.json`, and the hub block
  `keeplen` (section 4). It will send an interim note when the short lengths are scored.
- Prior evidence (4k-16k mixed pool, **[reported]**): against the recipe's efficiency at matched loss, keep
  10%/3.5% = 0.71x, 5%/2% = 0.55x, 2%/1% = 0.27x. `grJ` = 1.26x the recipe on the four-task mean but stalls
  on outlier (held-out CE 0.169 vs 0.10-0.13); the gain is contradiction plus extra steps.
- **When it reports:** check the per-length table against its result JSON and `ops`, relay the verdict
  (best keep per length, margin vs seed noise, which rule fits: constant fraction or constant kept-token
  count), fold its log entry into `experiment_log.json` (tracks `64k` and `mechanism`), confirm the hub
  block was filled.

### 2b. Early dense-loss proxy and the auto-research loop (PAUSED by Prasann 11-17 13:15)

- Session: "Early dense-eval-loss proxy for FLOP-drop methods", id
  `local_3a11446a-67b4-4997-8143-b67070f21dd5`, name `early-loss-proxy`. Worktree
  `.claude/worktrees/early-loss-proxy-d6e886`, branch `claude/early-loss-proxy-d6e886` (pushed). Prasann
  works with this session directly. **Do not archive it, do not resume the loop, do not republish its tab**
  until he says so.
- Its own state file is the authority: `research/token_routing/early_loss_proxy/LOOP_HANDOFF.md` on that
  branch (plus `README.md`, `BEAKER.md`, `LAMBDA.md`, `hub_tab/loop_state.json`).
- What it established:
  - Proxy check (record `records/early-dense-loss-proxy.md` on the branch, **[reported]**): step-10/20 dense
    held-out CE ranks 19 arms with Spearman 0.83-0.84 against their final 8k means; 30-50 steps give 0.95 on
    a 9-arm subset; steps 1-5 are uninformative; it cannot resolve final gaps under about 0.03-0.05. One
    check costs 50-215 s of 2-GPU time after a 49 s load (Prasann expected under a minute).
  - Auto-research loop, about 115 arms (**[reported]**, mostly one seed, held-out eval_size 184, well below
    500): best is the recipe on a pool that is 80% short rows (1k-2k) and 20% long rows (4k-16k), 3.1-3.7x
    fewer FLOPs than dense LoRA at matched held-out loss over a full epoch (two seeds), 3.2-3.5x on the 16k
    rows alone (56 rows) against 2.1-2.7x for the plain recipe. Position-preserving document subsampling
    (keep 25%) is 1.23-1.35x over the recipe at 16k. Negative: frozen prefix, earlier cuts, FFN skip,
    early-exit-loss selection, detached context, input-level token exit, drop-CPT bases, higher LR, 8
    rows/step, keep below 20%/7%, random keep, multi-question rows, loss-proportional sampling,
    document-local attention.
  - Memory (**[reported]**): transient slope 0.37 GiB per 1k tokens for dense LoRA (out of memory at 256k),
    0.24 for the recipe, 0.040 with early blocks in 4k chunks plus a 4,096 kept-token cap plus host offload;
    a 1M-token row trains on one 141 GB GPU. Learning with chunking at 64k+ is untested.
  - Validation with generation evals (**[reported]**, not published): recipe 3.3x fewer FLOPs than dense
    LoRA at matched score (8k 3.5x, 16k 3.2x) where the proxy said 1.9-2.2x; dense full fine-tune f1 0.928
    beats the recipe 0.897 and dense LoRA 0.882 (3-task mean; the reorder grader is broken). Treat dense
    full fine-tune as the stronger reference from now on.
- Unmerged loop branches: `loop/longpos` (flag `--longpos-docsub-keep`), `loop/validate` @ 787581912, and
  the per-idea branches `loop/{alloc,beaker,cptbase,cptnative,detach,ffnskip,goldrand,lambda,lenscale,prefix,rows}`.
  `loop/memscale` is merged into the proxy branch. About 57 git worktrees exist, most of them
  `.claude/worktrees/agent-*` from the loop's subagents; leave them until Prasann closes the loop.
- Jobs of this thread still alive: Beaker `elp-v3-bkZKTRFWVK` and `elp-v3-bkYMJCNZFX` (running, about 2 h
  in, expected 2.5 h); slurm `5201590`-`5201593` (`elp-v3`, pending over 1.5 h, three of them pinned to
  `sneetches`, which is unavailable). They count against the jsteinhardt 8-GPU `preemptive_high` cap if
  they start. Nobody has been asked to cancel them.
- **Pending on this thread:** its `experiment_log_entry.json` (track `mechanism`) is not yet in
  `experiment_log.json`; fold it in once Prasann and that session agree on a verdict.

### 2c. Hub reorganization (IDLE)

- Session: "Reorganize the FLOP-drop hub artifact", id `local_9a6d1692-d798-4df3-b6c6-023470949c95`, name
  `hub-reorg` (started by the proxy session). Worktree `.claude/worktrees/great-chatterjee-82e4e0`, branch
  `claude/great-chatterjee-82e4e0`.
- It regrouped the hub (Overview plus six groups) and, at my request on 11-17, cleaned up group 2 "Scale and
  length" and added a "Keep vs length" section (part id `keeplen`, markers `<!-- KEEPLEN:START/END -->`)
  at its top. It reported the page live as **version 117** [reported: I have not opened that version].
- Send hub layout work to this session; it has the context.

### 2d. Not mine, same machine

- "Beaker storage cleanup" (`local_77d4bf85-6219-4e8b-9941-910747b78b00`, idle since 11-15, main checkout,
  experiment `weka_cleanup`). Untouched by this session.
- The `pail_*` experiments in the index are run from other sessions that are not visible from here.

## 3. Finished since 11-09 (all in the hub log; branches pushed, none merged)

| thread | verdict | where |
|---|---|---|
| Rows <= 32k at matched FLOPs, 11-task mix (11-14) **[checked: dense f4 cells vs ops]** | 6-task means at 64k/128k/256k: <=16k data .506/.401/.324, <=32k .558/.452/.359, <=64k .581/.477/.393 at about 4.6-4.9k PF. Rows <= 32k recover 51-77% of the 16k-vs-64k gap but stay 0.02-0.05 under 64k data; the rest is mostly contradiction. 256k eval_size 125. | `claude/mt11-lengen-32k-56ed58` @ 809f27c63; `records/mt11-64k-length-generalization.md`; experiment `mt11_64k`; artifact https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n |
| Random non-gold document drop at recipe FLOPs (11-13) **[checked in ops]** | Negative. Best 8k mean 0.665 (group-aware outlier, original positions) vs recipe 0.846; outlier collapses to 0.00-0.42. FTM under 1.08x; best-case per-step 4.3-4.4x. | `claude/docdrop-nongold-42bb6b` @ cac6c3dd3; `records/docdrop-nongold-baseline.md` |
| Weighted token sampling (11-11) [reported] | Negative: sampled 0.828 vs top-k 0.847. Side result: recipe without the dense step 1 scores 0.848 at 24.2 GiB. | `claude/weighted-token-sampling-041da9` @ 7196f46d7 |
| Oracle threshold sweep (11-11) | Done; also added dense points at 488 and 976 rows/task (0.667 at 379 PF, 0.712 at 762 PF). | `claude/oracle-thresh-sweep-69bc7d` @ 178a6b346 |
| CPT gradient maps (11-10) | Per chunk-pair gradient share falls about 20x by 16k of distance after dense CPT and is flat beyond; a late-chunk oracle needs 20-31% of context tokens for cosine 0.95. Per-distance token sparsity was not measured. | `claude/cpt-gradmaps-34a768` @ 5187363ed; experiment `grad_composition` |
| Context-fidelity probe (11-10) | Done. | `claude/gallant-robinson-adc54a` @ 814ba80de |
| GPU audit (11-10) | 6,705 GPU-h from 10-20 to 11-10; 1,083 GPU-h (16%) lost to preempted Beaker restarts; about 60% of Beaker job lifetime pending. Known bug: the daily CSVs list 11-01 twice and drop 11-10 (DST), summing to 6,767. | `claude/gpu-workflow-audit-8833c4` @ e790ae6b2; experiment `gpu_audit`; hub section `gpuaudit` |

Reference numbers that keep coming up: recipe `lora64all32r20to7` 8k 4-task mean 0.846/0.847 at 1,952
rows/task vs dense 0.843, step cost 0.221x dense, 26.9 GiB peak, FTM 4.6x. Dense curve: 0.667 / 0.712 /
0.843 / 0.899 at 488 / 976 / 1,952 / 3,904 rows/task. Tail rule (fixed last part of each document, 11-04):
8k 0.781 (20%/7%) and 0.747 (10%).

Figures made for Prasann on 11-16: `research/token_routing/summary_figs/` (`make_summary_figs.py`,
`cpt_long_range_gradient.png`, `mt11_64k_length_generalization.png`; untracked).

The seven sessions behind the table were archived on 11-16; their ignored result files were copied to
`/accounts/projects/berkeleynlp/prasann/projects/OLMo-core-archive/worktree_files/<worktree>/` (the GPU
audit's `raw/` there holds a user map with real usernames).

## 4. The hub and the experiment log

- Hub: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (private). Source of truth is the **live page**,
  with a copy at `research/token_routing/ds64_fast8k/hub_live/hub.html`.
- Procedure (`hub_live/README.md`): read the live page first and build on that version; log refresh with
  `python3 hub_live/update_log.py HUB.html`; new section = append a `<div class="part" ... data-group=...>`;
  auto-research tab only via `hub_live/build_autoresearch_tab.py --state loop_state.json`;
  `node --check` the inline scripts; publish to the same URL; never `force`; never run `hub_live/legacy/*`
  or `make_overview_page.py` on the current page. Three sessions publish this page, and a republish from a
  stale copy already reverted a tab once (11-17).
- Log: `research/token_routing/ds64_fast8k/experiment_log.json`, 66 entries, newest first. Four entries from
  10-29 still say `running` and are stale (gradient concentration across stages; Qwen3 model-size ablation;
  recipe on Qwen3-4B; model-scale ablation): set their real status when you next touch the log. Side
  sessions leave an `experiment_log_entry.json` on their branch; the main agent folds it in. Status values
  are plain `done` / `dropped` / `running` (not `done (negative)`).
- Every experiment also keeps one artifact (global rule); `mt11_64k` has its own page (URL above).

## 5. Uncommitted state in the main checkout (nothing committed since 5e1eb35ca; commit only when asked)

- Modified: `CLAUDE.md`, `ops/experiments/INDEX.md`, `ops/experiments/{grad_composition,mt11_64k,tokdrop_lr}/STATUS.md`,
  `research/token_routing/ds64_fast8k/experiment_log.json`, `hub_live/README.md`, `hub_live/hub.html`.
- Deleted from `hub_live/` (moved to `hub_live/legacy/` by the reorg): `build_nov_tabs.py`,
  `patch_log_render.py`, `rebuild_hub.sh`.
- Untracked: `ops/experiments/gpu_audit/`, `ops/experiments/pail_pre2020/`, `ops/experiments/pail_weird/`
  (the two pail pages belong to other sessions), the new `hub_live/` files (`update_log.py`,
  `build_autoresearch_tab.py`, `reorg_hub.py`, `restyle_v3.py`, `verify_reorg.py`, `hub_v2*`, `hub_v3*`,
  `decisions.json`, `idea_impl.json`, `make_idea_impl.py`, `autoresearch_state.json`, `gpu_audit_data.json`,
  `REORG_PLAN.md`, `legacy/`), `research/token_routing/summary_figs/`, this record.
- `ops/experiments/tokdrop_lr/STATUS.md` was last closed out on 11-09; its body does not yet cover 11-10 to
  11-17. Run the `close-out-experiment` skill once the keep-fraction thread lands.

## 6. Decisions waiting on Prasann (ask; never act without an explicit OK)

1. Resume, redirect or close the auto-research loop; publish its validation result; what to do with its two
   running Beaker jobs and four queued slurm jobs.
2. Merge `claude/gpu-workflow-audit-8833c4` into `prasann/flopdrop_clean` (local eval launcher, minRuntime
   default, `ops capacity`).
3. Ceres as a default Beaker cluster: he said "ask me about this again later".
4. Recompute the hub's FTM multipliers with the new 488 / 976-row dense points.
5. Commit the uncommitted main-checkout edits (section 5).
6. mt11 next step: seeds, or a small long-row tail in the <=16k / <=32k mix.
7. Older items still open from 11-09: larger-data push, a new approach to blocks 0-6, merging side branches.

Small things noticed and not requested: the audit's DST double count in `analyze.py`; empty `artifact:`
headers in records on side branches; local re-score rows for oracle 0.90 / 0.80 not filed in `ops`; stale
slurm job 3832246 (`pail_classify`, pending 33 days, dependency never satisfied).

## 7. Rules that changed or were added since 11-09 (full text: `CLAUDE.md`, memory)

- Never merge into or push to `main` or the default branch without Prasann's OK, each time. Side work goes
  on its own branch, pushed.
- Beaker: allocated jobs are fine (the "always unallocated" rule is retired); priority exactly `urgent`;
  minRuntime = min(4 h, 1.3 x expected), never 0; runs over 4 h need a resume point.
- Small short jobs (about 2 GPUs, 2 h or less) spill to jsteinhardt when Beaker is backed up: at most 8 GPUs
  at `preemptive_high` across all sessions, more at `--qos=preemptive` for restartable work only.
  berkeleynlp is debugging only at `--qos=normal`. Lambda is usually busy: check `squeue` there first.
- Long loops must survive usage limits: a recurring heartbeat plus a state file, and rate-limited subagents
  resumed with SendMessage.
- Sessions: start side sessions with `mcp__ccd_session__start_session` (Opus 5.5, own worktree, fresh
  context, a full brief); send follow-ups to the session that has the context (`SendMessage` to its
  `local_...` id); archive only after Prasann agrees, never delete; copy ignored result files to
  `OLMo-core-archive/worktree_files/` before archiving.
- A side session's note is a report, not proof: check numbers against `ops` or the result files before
  relaying, and say which numbers are unchecked.

## 8. Traps hit in this stretch

- `cd` in a Bash call changes the session's working directory for later calls (I ended up inside a worktree
  twice). Use absolute paths or `git -C`.
- `archive_session` fails once with "a hook changed this call's input"; re-issue it with the hook-added
  `session` field.
- In `ops`, long-rung labels are `64k` / `128k` / `256k` (not `r64k`) and run names carry `-lr5e-06`.
- A per-distance gradient plot must be normalized per (target chunk, context chunk) pair; the raw share per
  1k of distance overstates the fall (about 100x instead of about 20x) because far distances have fewer pairs.
- Proxy memory values logged before 11-17 13:10 are wrong (the peak was reset before being read).
- Document-drop methods look fine on the mean and collapse on outlier (the kept gold outliers stop being the
  minority); always read the per-task and worst-task numbers.
- The proxy understated the recipe's gain (1.9-2.2x) relative to generation evals (3.3x); use it to rank and
  to reject, not to quote a speedup.
