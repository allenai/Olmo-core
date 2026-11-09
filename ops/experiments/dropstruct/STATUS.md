---
experiment: dropstruct
title: CPT objectives that make token dropping easy afterwards — token exit at layer L (rule / learned router) vs dense CPT, scored by tail CE under drop
status: paused
updated: 2026-10-14
branch: prasann/dropstruct (worktree /accounts/projects/berkeleynlp/prasann/projects/OLMo-core-dropstruct, local only, not pushed)
launcher: research/token_routing/dropstruct/driver_local.sbatch
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core-dropstruct/records/dropstruct-token-exit-cpt.md
results: ops result show --experiment dropstruct (152 rows); raw JSON /data/prasann/dropstruct/evals on horton + thidwick
wandb_group: https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/dropstruct-q35-4b
next_step: Prasann picks a direction from the record's ranked list; first = fix the plain-CPT regression (lr 1e-5 or 8+ rows/step) before scaling any arm
---

# dropstruct

## Question
Can a CPT objective make a model's context easy to drop afterwards? That is, does it concentrate
what later tokens need into predictable places (a rule) or cheaply predicted places (a router)?
Asked by Prasann on 2026-10-13: pick reasonable ideas from the brainstorm and run them overnight.

## Decisions
- 2026-10-13 (Prasann): try the reasonable brainstorm ideas overnight in a new worktree and report
  in the morning.
- 2026-10-13 (agent): one mechanism, **token exit** (`src/olmo_core/nn/token_exit.py`).
  - From block L on, the body tokens a compressed doc does not keep leave the sequence (compacted,
    original RoPE positions).
  - L=0 is input-level drop. L=8 lets kept tokens absorb dropped ones first.
- 2026-10-13 (agent): CPT = softdetach geometry on q35-4b-base-markerfix.
  - Local shard rebuilt from the public text sample, cpt80 eval docs excluded.
  - lr 3e-5, 64M tokens, 4 rows/step (2 GPUs).
  - Training: per-doc compress prob 0.5, rho ~ U[0.05, 0.5]; labels inside compressed docs ignored.
- 2026-10-13 (agent): router = linear logit on [RMS-normed block-L input, relpos], zero init,
  Gumbel top-k. It is trained through a straight-through coefficient on the kept tokens' K/V/beta.
  `routerA` anneals the noise from 1 to 0.
- 2026-10-14 (agent): added an in-distribution held-out dev shard (the stream continued past the
  training docs) after finding that plain CPT regressed cpt80.

## Current best numbers
Teacher-forced tail CE (no-cot). Deltas are paired vs the dense-CPT control on the same rows.
⚠ eval_size = 96 rows (89 for cpt80 at 32k); ± is the paired per-row SE. Seed noise from the dense
and L8fl replicates is ≤ 0.002 on the headline drop deltas and ≤ 0.004 on full CE.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| L8fl vs dense | cpt80 / 8k | ΔdCE L8-fl25 = −0.015 (dense +0.078) | 96 ⚠ | 0.001 | no-cot | ops result show (ev-cpt80-dscpt-L8fl-u64M) |
| L8fl vs dense | cpt80 / 8k | ΔdCE L8-rand25 = −0.034 (dense +0.136) | 96 ⚠ | 0.002 | no-cot | same |
| L8router vs dense | cpt80 / 8k | ΔdCE L8-rand25 = −0.071 | 96 ⚠ | 0.004 | no-cot | ev-cpt80-dscpt-L8router-u64M |
| L8fl vs dense | shard_dev / 8k | ΔdCE L8-fl25 = −0.017, L8-rand25 = −0.033 | 96 ⚠ | 0.002 | no-cot | ev-shard_dev-dscpt-L8fl-u64M |
| L0fl vs dense | cpt80 / 8k | ΔdCE L0-fl25 = −0.006 (input-level drop barely trainable) | 96 ⚠ | 0.001 | no-cot | ev-cpt80-dscpt-L0fl-u64M |
| L8router own model | cpt80 / 8k | router25 − rand25 = +0.018 (learned router worse than random) | 96 ⚠ | 0.002 | no-cot | summary_table.py |
| every drop arm vs dense | both / both | full-context CE +0.013 to +0.022 | 96 ⚠ | ~0.001 | no-cot | summary_table.py |
| dense vs base | shard_dev / 8k | full-context CE +0.020 (plain CPT regresses the base) | 96 ⚠ | 0.004 | no-cot | summary_table.py |

Reading:
- The L8 exit objective makes the model robust to late dropping in general: random selection
  halves its penalty.
- It does NOT concentrate information at rule positions. Every rule improves equally, and the
  rand − fl gap shrinks.
- The full-context cost roughly cancels the rule gain.
- Full write-up and next steps: the record linked above.

## Next step
Waiting on Prasann's direction. Ranked options are in the record:
1. Fix the plain-CPT regression.
2. Reduce the full-CE cost (lower compress prob / KL; check the KL against the 32k ban).
3. Force concentration (kept set as the only path).
4. Replace the straight-through router objective.
5. SFT + dev-loss grid downstream.

## Open issues / traps
- lorax GPUs are broken: `/dev/nvidia0` is missing and only 1 of 4 allocated devices initializes,
  but `where-to-run` shows "5 free". Don't use it.
- GPUs freed between chained jobs were re-taken within minutes (thidwick, horton). One driver
  allocation per node avoids this.
- `train_ctc_suite.py` marker audit (b3cc727ff) calls load_keys inside the process group, so
  every torchrun launch dies. This is a bug on flopdrop_clean, not fixed there. dropstruct runs the
  audit before torchrun and passes --skip-marker-audit.
- Token exit gives a new compacted length per forward, so FLA re-autotunes every time (~3x step
  time). Fixed in 8523999b8. The first drivers' staged train scripts were repointed to that commit
  at 00:18 (REV is logged per arm).
- `ctx.total_calls` in train_ctc_suite counts the whole shard, not --max-tokens. The token-exit
  anneal uses --max-tokens.
- Plain CPT regresses this base by ~0.02 nats even in-distribution: lr 3e-5, 4 rows/step.
  Unresolved.
- The loss divisor is counted before label masking (as in softdetach), so logged train CE is about
  0.5x on drop arms. Do not compare train CE across arms.
- Compacted GDN short conv sees the previous KEPT neighbour. This is the same in train and eval.

## Verified vs unverified
- Verified:
  - The shard build skipped all 192 cpt80 eval docs (dsbuild logs).
  - The dev shard starts past the training part (skip-docs 38602).
  - The marker audit passed on the staged base.
  - CPU invariants of the selection hold (check_token_exit.py, in the smoke jobs).
  - Every arm loaded the base (tolerant load: 531 keys).
  - Seed replicates agree.
- Settled 2026-10-16 (Prasann): the no-full-attention-at-32k directive is REVERSED. The L8 arms' full-context
  pre-exit blocks are fine, and a KL-to-full objective is no longer blocked by it.
