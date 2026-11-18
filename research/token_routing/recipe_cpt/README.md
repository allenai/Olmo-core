# recipe_cpt: continued pre-training of Qwen3.5-4B with the FLOP-drop recipe (2026-11-17)

**Question (Prasann, 2026-11-17).** Does continued pre-training (CPT) under the recipe itself make the recipe more
Pareto-optimal downstream? Concretely: CPT the base with the recipe at 32k for ~1B tokens and keep checkpoints at
125M / 250M / 500M / 1B tokens. From each checkpoint, run the recipe-vs-dense SFT comparison of the 8k primary screen
and ask whether the recipe's FLOPs-to-match-dense (FTM) gets better than from the original base. A dense
full-fine-tune CPT on the same rows is the control ("CPT data" vs "CPT with dropping"). Prasann approved it on 11-17.

State: `ops exp show recipe_cpt` (draft page: `STATUS_draft.md` here). Live jobs and paths: `HANDOFF.md`.

## Files

| file | role |
|---|---|
| `build_cpt32k_shard_beaker.sh` | Beaker CPU job: the `cpt32k_u1B` shard (softdetach's `build_cpt_shard.py --blocks 63`) |
| `launch_recipe_cpt.py` | the CPT runs (`recipe`, `dense`) and their smoke tests (`--smoke A` / `--smoke B`) via `beaker_ctc_suite.py` |
| `merge_lora_export.py` | checkpoint -> plain base: merges `W + 32 * B @ A` into every adapted linear (dense: a model-only re-save) |
| `eval_cpt_devloss.py` | softdetach's dev-CE evaluator plus `--lora-rank` (scores an unmerged checkpoint). It refuses a LoRA checkpoint scored without its adapters. |
| `post_ckpt_beaker.sh` | one 1-GPU job per checkpoint: export (merge) + full-context dev CE on the export |
| `launch_sft_from_cpt.py` | downstream: recipe and dense SFT on mt4 s1952 from an exported base, plus the 8k/2k evals |

## Design (decisions and reasons)

- **Code base: branch `claude/early-loss-proxy-d6e886` @ d52756c12.** The online router and its LoRA flags
  (`--st-otr-stages`, `--st-otr-early-lora*`) are not on `prasann/flopdrop_clean` at all. They live on
  `claude/online-router-1bc01a`, and the proxy branch descends from it. The proxy branch adds about 1.7k lines of
  router, FLOP-meter and checkpointer work: router state in the trainer state for resume, the ephemeral save
  interval, and the memory work. The keep-fraction session also cut from this branch. The loop's paused run-state
  files are on the per-idea `loop/*` branches, not here.
- **Base: `ctc_suite/bases/q35-4b-base-markerfix`.** This is the marker-repaired base every recipe and dense SFT run
  of the 8k screen starts from, so downstream comparisons are like for like. Recipe CPT freezes the embeddings, so the
  repaired marker rows carry over unchanged (`merge_lora_export.py --base` reports it). The dense control trains them.
- **Data: `recipe_cpt/shards/cpt32k_u1B`**, 31,008 rows x 32,257 tokens = 1,000,225,056 tokens, 996,287,040 loss
  tokens.
  - It is the softdetach `cpt_u1B` stream: amandab's tokenized dolma3_longmino 15B sample, source parts 0-27, cut
    into 510-token bodies wrapped in `<|doc_start|>`/`<|doc_end|>`. Rows here hold 63 pseudo-docs instead of 127, so
    they fit a 32k window (`--seq-len 32768`, 511 pad tokens per row).
  - The build job checked that the source has 29 parts, so the dev part (28, softdetach's `cpt_dev`) is never read
    by training.
  - Same pseudo-doc geometry as softdetach, so the recipe's per-document allocation acts on the 512-token blocks.
- **Recipe: arm `lora64all32r20to7` unchanged.**
  - Whole-model LoRA r64, scale 32. Base weights and embeddings are frozen; the LM head trains at full rank, as in the
    SFT recipe.
  - Online ridge router, keep 20% of each pseudo-doc's body after block 7 and 7% after block 16. Explore 0.1,
    `--st-otr-doc-share 1`, `--st-otr-dense-every 0` (no dense steps).
  - Two CPT-only flags, both from olmo3_cptdrop's CPT router runs: `--st-gold-blind` (CPT rows have no gold
    documents) and `--st-otr-no-query` (no query, so no h*q features).
- **What the recipe does to the CPT loss (read before interpreting).** Dropped tokens leave the stack, so they get
  **no LM loss**.
  - Per row, only the ~7% of body tokens that survive both stages are trained as targets. The router picks them
    (high predicted gradient norm), plus the markers.
  - Their labels are the original next tokens, and they attend to every token through blocks 0-6 and to the 20% set
    through blocks 7-15.
  - So 1B tokens of recipe CPT trains ~70M targets. A dense step trains 1B.
  - olmo3_cptdrop measured exactly this family on full-context dev CE and it lost: online router keep 10% @ block 7,
    m ~0.6-0.8; router keep 0.5 on top of whole-model LoRA, worse than 4 dense steps. **A dev-CE loss here is
    therefore expected and is not the question.** The question is downstream SFT.
- **LR 2e-5, 16 rows/step (516k tokens), WSD (3% warmup, constant, decay over the last 10%), both arms.**
  - olmo3_cptdrop found 2e-5 best at 524k tokens/step, for dense and for whole-model LoRA alike.
  - Plain CPT regressed in dropstruct at 3e-5 and 4 rows/step; softdetach's dense CPT used 3e-5 at 8 x 64k.
  - Same LR for both arms, per Prasann (no measured reason to differ).
  - WSD keeps the 125M / 250M / 500M checkpoints at the constant LR, so they are comparable to each other; only the
    1B point is annealed. Annealed intermediate bases would need short decay branches
    (`train_ctc_suite.py --branch-from <stepN> --max-tokens ...`); not done.
- **Steps and checkpoints.** 1,938 steps. Permanent checkpoints every 242 steps (`--max-checkpoints 0`); the 242 /
  484 / 968 / 1936 checkpoints are the 125M / 250M / 500M / 1B bases (124.9M / 249.8M / 499.6M / 999.2M tokens). The
  others (726 / 1210 / 1452 / 1694) are extra resume points and can be deleted after the run. Ephemeral resume
  checkpoints every 121 steps. The post-fit export (`<run>/model_and_optim`, step 1938) is the final 1B base.
- **Resume.** A Beaker preemption restart keeps the launch id, so `run_guards.check_run_identity` lets it resume from
  the newest checkpoint; the router fit and FLOP meter are in the trainer state. A manual relaunch into the same
  folder needs `--resume` (refused otherwise: the silent auto-resume guard). Smoke B tests that path.
- **Checkpoints -> SFT bases.** `merge_lora_export.py` merges the adapters into the base weights in fp32 and writes a
  plain model-only `model_and_optim` with exactly a plain Qwen3.5-4B's keys. It loads strictly into the dense arm and
  tolerantly into the recipe arm, like the base it replaces. The smoke checks merged dev CE == unmerged dev CE. Dense
  control checkpoints get a model-only re-save (drops ~40 GB of optimizer state).
- **Reads queued per checkpoint** (`post_ckpt_beaker.sh`):
  - Full-context dev CE on `cpt_dev`, plus the 0-32k / 32k-64k position bins. ⚠ eval_size=32 rows (~2M tokens).
  - The recipe's own-construction CE cannot be rebuilt offline (the router's ridge fit lives in the trainer state).
    The run's own-construction number is the training CE on its kept tokens (wandb `train/CE loss`).

## Smoke and probes (2026-11-17, Beaker jupiter, 8 x H100, production geometry)

Throughput, memory and FLOPs (steady state, steps 10-20):

| arm | s/step | tokens/s/GPU | peak active / reserved GiB | metered FLOPs/step vs dense | 1B tokens (1,938 steps) |
|---|---|---|---|---|---|
| recipe | 3.30 | ~20,000 | 19.1 / 26.7 (23.0 on the dense warmup step 1) | 0.179x (cumulative 0.207x over 20 steps, incl. the dense step 1) | ~114 min, ~15 GPU-h |
| dense | 8.38 | ~7,820 | 31.2 / 32.4 | 1x (15.6 PF/step) | ~279 min, ~37 GPU-h |

Wall-clock speedup 2.5x against a 5.6x per-step FLOP cut: at 32k, attention and the frozen-weight forward are not cut
by the drop.

Checks that passed:
- **Router.** Stage 1 keeps 20% of each pseudo-doc (`mid_row_frac` 0.203); stage 2 keeps 35% of the survivors
  (`s1_mid_row_frac` 0.366), so 7.4% of the row reaches block 16+. Router rank correlation 0.37-0.42.
- **Resume** (smoke B, `--resume` into the step-20 folder): loads step 20; data loader resumes at batch 20; router
  fit resumed (2 stages); FLOP meter accumulators restored; steps 21-30 continue at the same CE level. A Beaker
  preemption restart uses the same path without `--resume`.
- **Checkpoints.** The async ones cost ~5 s (recipe) and ~7 s (dense); the synchronous one ~11 s / ~23 s.
- **Merge.** Merged export vs unmerged checkpoint: dev CE 1.3685 vs 1.3686 (8 rows). Embeddings unchanged;
  `||32 B A|| / ||W||` = 2.6% mean after 20 steps at 2e-5.
- **Base.** Dev CE 1.3199 ⚠ eval_size=32 rows (softdetach measured 1.320).

**Trap found.** `beaker_ctc_suite.py` always passes `--no-final-checkpoint`, and the trainer then disables the
checkpointer unless `--ephemeral-save-interval` is set. `--save-interval` alone writes nothing: the first smoke ran
20 steps with `--save-interval 10` and saved no checkpoint. The launcher always passes both.

**Finding: recipe CPT raises full-context dev CE immediately, and LoRA is not the cause.** Setup: 20 steps (1-step
warmup) of each arm, then full-attention dev CE on the first rows of `cpt_dev` (no-cot, teacher-forced).
⚠ eval_size=8 rows, so these are a direction check, not a measurement.

| arm (20 steps) | CE, mean of rows 1-5 (paired) | Δ vs base | CE, 8 rows |
|---|---|---|---|
| base | 1.1208 | | (1.3199 on 32 rows) |
| dense full FT, LR 2e-5 | 1.0953 | -0.026 | 1.1802 |
| whole-model LoRA r64, no routing ("loradense"), LR 2e-5 | 1.0802 | -0.041 | 1.1661 |
| recipe, LR 5e-6 | 1.2367 | **+0.116** | 1.3228 |
| recipe, LR 1e-5 | 1.2918 | **+0.171** | 1.3734 |
| recipe, LR 2e-5 | 1.2877 | **+0.167** | 1.3685 |

- LoRA at 2e-5 without routing beats dense, so the regression is the routing: training only ~7% of targets, on
  contexts compacted after blocks 7 and 16.
- The regression grows with LR and plateaus between 1e-5 and 2e-5.
- This is the olmo3_cptdrop failure (every router arm lost on full-context dev CE), stronger here.
- The kept-token training CE is flat over 30 steps. The logged `train/CE` is the kept-token CE sum divided by ALL
  label tokens: about 0.24 = ~3.2 nats on the 7.4% kept targets.
- The run was launched anyway, as designed. It is cheap (~15 GPU-h), and the question is downstream SFT, not CPT
  dev CE. Read each checkpoint's dev CE before scheduling SFT from it.

## Prior drop-CPT bases (why this is not a repeat)

The paused auto-research loop tested SFT from drop-CPT bases on its proxy (`loop/cptbase`, `loop/cptnative`) and saw
no gain. Those bases were dropstruct's 64M-token CPTs:

- `dscpt-dense-u64M` (plain CPT);
- `dscpt-L8fl-u64M` and `dscpt-L0fl-u64M`: token exit at layer 8 / 0 under a fixed first+last rule, compress prob
  0.5, keep rho ~ U[0.05, 0.5];
- all at LR 3e-5, 4 rows/step, 64k rows. dropstruct found plain CPT at that setting REGRESSED the base.

`cptnative` arms were short in-loop variants of the same rule. None of those bases was CPT'd with the recipe itself
(LoRA + online two-stage router). None ran past 64M tokens. None used the recipe SFT's per-document 20%/7% budget.
This run uses the recipe as is, 16x more tokens, the olmo3 LR/batch, and a matched dense control.

## Storage

- Shard: `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/recipe_cpt/shards/cpt32k_u1B` (4.0 GB).
- Runs: `/weka/.../prasanns/ctc_suite/ckpts/rcpt-q35-4b-{recipe,dense}-u1B/step*`.
- Exported bases: `/weka/.../prasanns/recipe_cpt/bases/<run>-step<N>/model_and_optim`.
- Dev CE JSONs: `/weka/.../prasanns/recipe_cpt/devloss/`.
