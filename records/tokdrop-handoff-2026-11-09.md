---
title: Handoff — tokdrop_lr main agent (token-routing recipe for Qwen3.5-4B CTC long-context SFT), state at 2026-11-09
date: 2026-11-09
updated: 2026-11-09
experiment: tokdrop_lr
area: token_routing
kind: handoff
status: current
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt
summary: Start here to take over tokdrop_lr as the main agent. The recipe (whole-model LoRA + online per-document router, 20% of document tokens after block 7, 7% after 16) trains Qwen3.5-4B to dense quality at ~4-4.6x fewer FLOPs (8k rung), and every attempt to reach 5-10x has failed. Covers the goal, where state lives (STATUS.md, experiment_log.json, hub source), results so far, what is ruled out, the mechanism picture, open options, working rules and traps. Nothing is running.
---
# Handoff: tokdrop_lr main agent (2026-11-09)

You are taking over as the **main agent** for Prasann's token-routing work. Side sessions did most
experiments; the main agent plans, launches side sessions or subagents, checks their numbers, and keeps the
experiment log, the hub and STATUS.md current. **Nothing is running now** and every side session is
archived.

## 1. The goal

Train Qwen3.5-4B (hybrid: full attention + Gated DeltaNet, 36 blocks) on CTC long-context SFT data
(4-task "fast8k" mix: contradiction, nq, outlier, oolong; rows up to 12,288 tokens) to dense quality with
far fewer training FLOPs. Inference stays dense over the full context. Prasann's bar (10-29):
**≥5-10x dense FLOPs-to-match (FTM)**; 1.25x tweaks are too small. Methods must hold at large data and must
be implementable fast at long lengths (FLOPs first, wall-clock later).

**Primary metric (Prasann 11-02): the 8k rung, 4-task mean.** 2k is secondary (near ceiling). FTM = how
many times more training FLOPs dense needs to reach the method's 8k score, read off the dense curve
(dense LR 5e-6 at 1,952 / 3,904 / 5,856 / 7,808 fresh rows per task).

## 2. The recipe (current best)

`lora64all32r20to7` (run tag; launcher `research/token_routing/ds64_fast8k/launch_fast8k.py`):
- Whole-model LoRA r64 (scale 32), LR 1e-5, 16 rows/step.
- Online token router: a ridge regression refit every step on per-token gradient-norm targets from the
  training backward (layer-4 features). Per-document allocation (`--st-otr-doc-share 1`).
- All tokens run blocks 0-6. After block 7 keep 20% of each document's body tokens; after block 16 keep 7%
  (nested; dropped tokens leave the stack). Prompt, markers, query and answer are always kept. Original
  position ids.
- 8k 4-task mean at 1,952 rows/task: recipe 0.845 / 0.847 (2 seeds) vs dense 0.843. 8k-only FTM:
  **4.6x / 3.6x / ≥4.5x** at 1,952 / 3,904 / 7,808 rows/task (STATUS.md has the table and the 8-cell view).
- Random selection at the same budget nearly ties the router at larger data (the router's edge is on outlier
  and shrinks with data).

## 3. Where everything lives

| what | where |
|---|---|
| Canonical state | `ops/experiments/tokdrop_lr/STATUS.md` (`ops exp show tokdrop_lr`) |
| Experiment log (standing rule: every session keeps it current) | `research/token_routing/ds64_fast8k/experiment_log.json` — newest first; entries may carry `tables`, `chart`, `figs` |
| Summary artifact ("the hub", private) | https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (v97). **Source: `research/token_routing/ds64_fast8k/hub_live/hub.html`** — see its README |
| Numbers | `ops result show --experiment tokdrop_lr [--json]`; never quote a number that did not pass `ops result add --check` |
| Runs / jobs | `ops status`, `ops stall <job>`, `ops runs --experiment tokdrop_lr` |
| Records (one per side experiment) | `records/*.md`; most live only on their side branches (table below) |
| Archived worktree files (results, launch logs) | `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core-archive/worktree_files/<worktree>/` |
| wandb | group `f8k-lra-q35-4b` in prasanns-allen-institute-for-ai/memory-networks |

**Updating the hub.** The live hub has drifted from `make_overview_page.py` (the generator lacks the
Gradient maps, Selection ablations and Prior work tabs). Do not regenerate from it. To update: edit
`experiment_log.json`, then splice the log into `hub_live/hub.html` (`const D_LOG = …; const D_COMPUTE`,
escape `</` as `<\/`) and republish to the same URL. `hub_live/rebuild_hub.sh` rebuilds from the v95 base
and the build scripts (adjust its scratchpad path first; the v95 base file was in the old session's scratchpad,
so prefer editing `hub_live/hub.html` directly). Per-experiment tab sections are `<div data-logfig="<log title
prefix>">` blocks that render the entry's text, chart and tables, so updating a log entry updates its tab.
Check rendering with headless Chrome (`google-chrome --headless=new --screenshot=… "file://…/hub.html#selection"`).

**Side branches (all pushed, none merged into `prasann/flopdrop_clean`; merging is pending Prasann):**

| branch | tip | what |
|---|---|---|
| claude/online-router-1bc01a | 8ce5fd22a | online router code; Beaker -tde runs pinned at 67a28d605 |
| claude/otr-perrow-minfloor | 315f37b03 | per-row budget + floor; record perrow-floor-recipe-ablation.md |
| claude/cross-row-alloc | 7344c21ec | cross-row threshold, selective backprop; record cross-row-allocation.md |
| claude/posid-reset-0abac3 | d121b6f6e | position-id reset + tail rule; records posid-reset-…, tail-rule-… |
| claude/oracle-train-bd7414 | 8ec4fdb14 | oracle-selection training (`src/olmo_core/nn/oracle_token_select.py`) |
| claude/grad-support-tokens | 8eca2e734 | whole-token oracle under routing; gradient maps (log + linear) |
| claude/grad-concentration-d91a37 | 9bfb5313b | gradient concentration by stage |
| claude/grad-routing-gap | c5bb14145 | attribution vs routed gradient; early-phase probe |
| claude/mt11-lengen-bed337 | c6326240d | 11-task 64k length generalization to 128k/256k; 16k-capped data |
| claude/lit-review-efficient-lc-eacf2d | eeccbfb43 | literature review |
| claude/tokdrop-decay-fix, sparse-backward, multiview-h7cache, recipe-scale-ablation-71d007, qwen3-recipe-c876d1, vibrant-williams-4b0f66, train-layerskip-35010b | (10-28..10-30) | earlier side experiments, see STATUS.md |

## 4. What has been tried (all on the hub; 1,952 rows/task, 8k mean, eval_size 500/cell, 1 seed unless noted)

Toward 5-10x, all negative or ties:

| idea | result |
|---|---|
| Aggressive keep 10%→3% / 5% | 3.1x / 2.4x FTM at 1,952; ties at 3,904; blocks 0-6 are a dense floor (~0.2x/step) |
| Sparse backward, multi-view from cached block-7 state, KV-only, layer skip, late widening, ReLoRA | all trail or within noise |
| Per-row budget (2 seeds) / + floor 1 / 2 | 0.843 / 0.831 / 0.840 vs recipe 0.846; 3,904: 0.880 vs 0.882/0.889 |
| Cross-row step threshold / selective backprop top / sample 30% (2 seeds) | 0.832 / 0.684 / 0.739 |
| Position-id renumbering after each cut | 0.820 (3,904: 0.870 vs 0.886); outlier only |
| Fixed per-document tail (recipe shape / last 10%) | 0.781 / 0.747: the router's choice matters |
| Oracle-selection training (dense backward picks tokens each step; 2.3-3.4x cost/step, upper bound) | per-doc 0.850 (35% kept); per-row 0.674; batch 0.756; batch+random 0.806; grown to routed cos 0.95 0.837 (71% kept) |
| Model scale | Qwen3.5 gain grows with size (0.8B 1.7x … 9B 4.8x); Qwen3 flat ~2-3x |
| Length | 11-task 64k: recipe gain persists to 128k/256k (+0.046/+0.033/+0.054 at 64k/128k/256k, ⚠ eval_size 125 at 256k) |

## 5. Mechanism picture (what we believe and why)

- **Gradient attribution under the dense forward is not a proxy for routing.** Token sets holding 95% of the
  attribution (0.5-20% of tokens depending on scope) give a true routed gradient cos of only 0.08-0.19 once
  the other tokens leave the forward.
- **The recipe works through co-adaptation.** On the router-trained model the routed gradient is nearly
  unbiased (expected cos 0.87) from about step 31; random selection on that model stays biased. Gradient-
  faithful token sets need ~70% of the body; per-document allocation is the only allocation that survives.
- **Where the gradient lives.** Answer + query tokens carry 60-95% of blocks 7-31 energy; gold documents 8-55x
  distractors per token; document gradient sits in blocks 0-15 and a few tokens (linear-scale gradient maps).
  Token rank correlation between layers: 0.57-0.61 (block 7 vs 16), 0.33-0.39 (7 vs 31).
- **Example-level sparsity.** By 7,808 rows/task, 78% of rows are essentially solved and the top 10% of rows
  hold 81% of the answer loss. A token router with a per-document quota cannot use this; a method choosing
  which rows to spend compute on could (the selective-backprop arms tried a crude version and lost).
- Router ranking accuracy: Spearman 0.51-0.67 vs its own target. Never measured: the trained router's picks
  vs the true gradient at blocks 16/24/31 (its `rank_corr_explore` / `rank_corr_kept` are logged in wandb,
  not pulled).

## 6. Open options (none started; Prasann has not chosen)

1. **Larger-data push** (15,616 rows/task, 11-task, 32k-64k rows) — pending Prasann.
2. **A new approach to blocks 0-6** (they see every token and set a ~0.2x/step floor) — pending Prasann.
3. **Merge the side branches** into `prasann/flopdrop_clean` — pending Prasann.
4. Offered, no answer: pull the router's `rank_corr` logs and run a per-layer router-vs-true-gradient check
   (short 1-GPU job on jsteinhardt).
5. Offered, no answer: the two-cut batch-oracle + random arm `orcbatchr2` (defined in
   `research/token_routing/oracle_train/launch_oracle.py`, never run).
6. Literature-review baselines worth running: Random-LTD per-layer resampling, attention-score selection at
   block 7 (LeMo / A-MoD style), an end-to-end MoD / TokenGate gate, LongLoRA S2-Attn, and InfoBatch / GRESO-
   style forward skipping of solved rows (the review's most promising route to 5x+).

Ask Prasann before starting any of these.

## 7. Working rules (from CLAUDE.md and memory; read those for the full text)

- **Where to run:** debugging/smokes on jsteinhardt or Lambda (`preemptive_high`, up to 8 GPUs each, Rhys's
  jobs count within the 8); training, sweeps and eval waves on Beaker at priority `urgent`, minRuntime `1h`,
  workspace `ai2/flex2` (unallocated). berkeleynlp = small debugging only, `--qos=normal`. A100s: Lambda and a free balrog are pre-approved; ask
  before other A100s (Beaker saturn/prometheus); nothing below A100.
- **Every launch:** `ops register` (skill `launch-run`). Check jobs with `ops status` / `ops stall`, never
  sleep/poll loops.
- **Every number:** `ops result add --check` before quoting (skill `record-results`); `n` = corpus size,
  write `eval_size`; flag anything under 500 eval examples inline with ⚠ and its size; label cot.
- **Every experiment:** a log entry in `experiment_log.json` (with tables/chart when there are numbers) and
  its place on the hub; republish to the same URL.
- Subagents: pin `model` to sonnet or opus. Never push or merge to `main` without asking. Commit only when
  asked. Never add citations when editing Prasann's text.
- Before any document-chunked / landmark training from a fresh base, repair marker embeddings
  (`recipes/data/fix_marker_embeddings.py`).

## 8. Traps hit in this stretch

- **FTM below the first dense point is an upper bound (`<=`), not `>=`.** The per-row collector printed `>=`
  before the 11-04 fix (fixed in `collect_perrow_floor.py`); posid-reset is `<=4.46x`, floors `<=4.55x`.
- **Run-tag confusion:** `ds0` = per-row allocation, `pr` = position-id reset, `t10` / `t20to7` = tail rule,
  `rnd` = random selection, `sdN` = seed N, `fNNNN` = fresh rows/task, `sNNNN` = original data;
  `tds08b` / `tds2b` / `tds9b` = Qwen3.5 0.8B / 2B / 9B scale arms, `q3*` = Qwen3 arms. I once read the
  `pr` 3,904 result as per-row; check the tag against the launcher before quoting.
- **Evals can finish without being collected.** Eval jobs launched by some launchers register under
  `ds64_fast8k`, not `tokdrop_lr`; the per-row 3,904 evals sat uncollected for 6 days. When a training run is
  done, check `ops result show --run-id <run>` has 8 rows (4 tasks x 8k/2k).
- The hub generator has drifted from the live page (above). `cp` is aliased to `cp -i` here; use `\cp -f`.
- `ops status --record` across all experiments can exceed 2 minutes; run it in the background or with
  `--experiment`.
