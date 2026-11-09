---
name: close-out-experiment
description: >
  Use this skill before clearing or compacting a session, when finishing a chunk of work on an
  experiment, when handing off ("wrap up", "save state", "we're done for today", "write a
  handoff", "/clear", "/compact"), and when the user switches to a different experiment. Writes the
  experiment's current state to its STATUS.md, regenerates the index, records final run states
  and cleans up background watchers, so the next session starts from the truth.
---

# Close out an experiment

`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops`. One session works on one experiment; close it
out before moving on.

1. **STATUS.md** (`ops/experiments/E/STATUS.md`; create with `$OPS exp new E`): set `status`
   (active / paused / done / abandoned), `updated: <today>`, `next_step` (one line). In the body:
   - **Decisions:** dated, who decided (`2026-10-12 (Prasann): ...`).
   - **Current best numbers:** with eval_size, ±SE, cot label and source path (only numbers that
     passed the `record-results` skill).
   - **Open issues / traps** and **Verified vs unverified**: mark anything you didn't check against a
     primary source as `UNVERIFIED`.
   - Link any new records/ doc in `records:`.
   - **Artifact** (default for every experiment, Prasann 2026-10-20): publish or update the
     experiment's one summary artifact (verdict first, setup, charts from the real numbers, key tables
     with eval_size flagged, limits, source paths; include work done in side sessions, marked
     in progress if still running). Republish to the URL already listed in `artifacts:` rather than
     making a new page; put a new URL in `artifacts:` and in each record's `artifact:` header line.
2. `$OPS exp index` (regenerates `ops/experiments/INDEX.md`); `$OPS exp check` must be clean.
3. **Runs:** `$OPS status --experiment E`, then `$OPS update RUN final_state=done|failed|cancelled`
   for each finished run (or `$OPS status --record`). Register any launch that is missing.
4. **Snapshot live state:** `git -C /accounts/projects/berkeleynlp/prasann/ops_state add -A && git -C
   /accounts/projects/berkeleynlp/prasann/ops_state commit -qm "snapshot: <experiment>"` (runs + results live
   there, outside any code checkout; skip if nothing changed).
5. **Clean up:** stop the Monitor tasks, background shells and watcher loops you started; leave
   real jobs running and say which are still live.
6. **Memory:** only durable rules (a new trap, a directive from Prasann, an env fact). **Never put
   experiment state, numbers or "in flight" lists in memory**: those go in STATUS.md. If a memory file
   is now covered by STATUS.md, add `superseded_by: ops/experiments/E/STATUS.md` to its description
   and drop it from the MEMORY.md index.
7. Tell the user: what changed, what is still running, the next step.
