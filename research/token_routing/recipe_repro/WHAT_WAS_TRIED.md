# What was tried around the FLOP-drop recipe (state at 2026-11-17)

The recipe is `lora64all32r20to7`. It trains Qwen3.5-4B with whole-model LoRA r64 (scale 32, all 32 blocks, LR 1e-5)
and an online ridge router, refit every step on per-token gradient-norm targets. After block 7 the router keeps 20% of
each document's body tokens, and after block 16 it keeps 7% (`--st-otr-stages 7:0.2,16:0.07`, per-document
allocation, original position ids). Reference: 8k 4-task mean 0.846 (2 seeds, 0.845/0.847) vs dense 0.843 at 1,952
rows/task, step cost ~0.22x dense, 8k-only FLOPs-to-match (FTM) 4.6x / 3.6x / >=4.5x at 1,952 / 3,904 / 7,808 rows/task.

**How to read the tables.**
- Unless a row says otherwise, the setup is 1,952 rows/task (f1952) on the 4-task mix (contradiction, nq, outlier,
  oolong), 1 seed, eval_size 500 per task x rung, and no-cot.
- "8k mean" is the 8k-rung 4-task mean, the primary metric. "8-cell" is the mean over 8k+2k, an older headline that
  some rows still use (recipe 8-cell 0.883 / 0.915 / 0.924 at 1,952 / 3,904 / 7,808).
- FTM is how many more FLOPs dense needs to match the score. "FTM <=" is an upper bound: the arm scores below the
  lowest dense point.
- `n` always means documents per row (corpus size), never eval-set size.
- Seed noise on the 8k mean is about ±0.01-0.02 (sampled arms ±0.02), so single-seed differences under ~0.02 are ties.

**Provenance marks.**
- **[proxy]** = from the paused auto-research / early-loss-proxy loop: proxy-only, mostly one seed, ⚠ eval_size 184
  (teacher-forced held-out answer CE under the dense forward, pools of 4k-16k rows). The proxy understated the recipe's
  gain against generation evals (proxy 1.9-2.2x, generation evals 3.3x), so use it to rank and reject ideas, not to
  price them. Its "Nx" figures are FLOP ratios at matched held-out loss, against dense LoRA or against the recipe.
- **(reported)** = a side-session claim the main agent has not checked against `ops`.
- ⚠ = an eval or probe with fewer than 500 examples.

**Pointers.**
- `log: "..."` is a title prefix of an entry in `research/token_routing/ds64_fast8k/experiment_log.json` (main
  checkout, branch `prasann/flopdrop_clean`).
- A `records/` path with no branch exists in this worktree. Otherwise the branch named holds it.
- `[proxy]` rows point to `research/token_routing/early_loss_proxy/hub_tab/ideas.json` (by `id`) and
  `LOOP_HANDOFF.md`, both on branch `claude/early-loss-proxy-d6e886`, plus the per-idea branch `loop/<id>`
  where one exists.
- Hand-offs with full context: `records/tokdrop-handoff-2026-11-17.md` and `records/tokdrop-handoff-2026-11-09.md`.
  Experiment state: `ops/experiments/{tokdrop_lr,learned_router,mt11_64k}/STATUS.md`.

## 1. Selection rules (which tokens survive)

| idea | verdict | status | pointer |
|---|---|---|---|
| Random token choice at the recipe budget | Nearly ties the router: +0.027 for the router at 1,464 rows, a tie at 3,904; 0.914 vs 0.915. The router's edge is mostly outlier and shrinks with data. [proxy]: random keep reaches 0.54x the recipe's efficiency. | neutral | log: "Review controls"; log: "Small-model front end" |
| Fixed per-document tail (keep the last part of each doc, no router) | 8k 0.781 (recipe shape, 20%/7%) and 0.747 (last 10%) vs 0.846. Outlier carries most of the gap. The model does not just adapt to any drop shape. | negative | records/tail-rule-recipe-ablation.md (branch claude/posid-reset-0abac3) |
| Router target ablation | The gradient-norm target is best. A residual-change (activation strength) target is near random. | works (as adopted) | log: "Split-point sweep"; records/online-token-router.md |
| Exploration share (uniform-random part of the budget) | 0-50% all fine. | neutral | log: "Split-point sweep" |
| Router hill-climb: threshold keep, MLP router, doc-context features, task one-hot, per-task / per-group routers, keep schedule | Everything ties the linear router (0.83-0.86 vs 0.852 on the 1,464-row screen) or loses. A better gradient predictor (rank corr 0.54 vs 0.37) does not score better. MLP + per-group ridge replicate: 0.856, no gain. | neutral | log: "Router hill-climb round"; log: "Hill-climb replicates" |
| Per-row budget (vs per-document) and per-doc floor 1 / 2 tokens | 2-seed 8k 0.843 / 0.831 / 0.840 vs 0.846. At 3,904: 0.880 vs 0.882/0.889. Per-row concentrates on gold docs (leak_auc 0.73-0.75 vs 0.48) without helping. | neutral | records/perrow-floor-recipe-ablation.md (branch claude/otr-perrow-minfloor) |
| Cross-row allocation: one keep threshold per step | 2 seeds: 0.832 vs 0.846 at 1,952. 0.891 vs 0.885 at 3,904 (noise). | neutral | log: "Cross-row allocation"; records/cross-row-allocation.md (branch claude/cross-row-alloc) |
| Selective backprop (backward only on high-loss rows) | 30% of rows at 1,952 loses (0.68 top / 0.74 sampled vs ~0.78 recipe at equal PF). Top 50% at 3,904: 0.876 at 0.74x PF (within noise). The router barely ranks rows (Spearman 0.1-0.3). Untested at 7,808. | negative at small data | same as above |
| Weighted token sampling (sampled survival, w=1/p in attention / DeltaNet, price rule) | Never matches top-k. 9 sampled runs average 0.828 (sd 0.019) vs 0.847 for top-k. 3,904: 0.877 vs 0.882/0.889. At half the keeps: 0.805 vs 0.821. | negative | records/weighted-token-sampling.md (branch claude/weighted-token-sampling-041da9) |
| Oracle threshold sweep (per-doc attribution-cos 0.90 to 0.60) vs matched random | Oracle at about 18% keep ties random (0.824 vs 0.813/0.841). At about 12% random wins (0.796 vs 0.721; outlier 0.740 vs 0.409). Best oracle FTM is 4.16x, below the recipe's 4.6x. Oracle picks pile onto each document's first tokens. | negative | records/oracle-threshold-sweep.md (branch claude/oracle-thresh-sweep-69bc7d) |
| Oracle-selection training (dense backward picks the 0.95-cos set each step; costs ~2.3x dense/step, upper bound) | Only per-doc survives: 0.850 at 35% kept. Per-row 0.674, batch 0.756, two-cut per-doc 0.716. Growing the set until the routed cos reaches 0.95 needs 71% of tokens (0.837). | negative (as a method) | records/oracle-selection-training.md; log: "Oracle-selection training" |
| Mixture-of-depths gate (capacity 0.25, alternating blocks 8-30) | Loses outlier (-0.25) at 0.78x FLOPs. | negative | log: "Mixture-of-depths" |
| Selection by an early-exit answer-loss gradient [proxy] | Worse than the ridge router at the recipe keep (0.213 vs 0.175 held-out CE). Wins only at a block-2 cut, far from any reference level. | negative | [proxy] id=earlyloss; loop/earlyloss |
| Small-model front end (0.8B picks tokens for the 4B) | Cost model only: ceiling ≈1.35x over the recipe even with no score loss; realistic ≈1.9x dense. Selection is not the bottleneck. | negative (not trained) | log: "Small-model front end"; branch claude/small-model-select-0a941a |
| Earlier frozen / offline routers and fixed keep heuristics (gold+first/last, devloss grid) | Superseded by the online router. | superseded | records/learned-token-router.md; records/devloss-grid-underperformers.md; log: "Earlier compute-saving work" |

## 2. Where and when to cut

| idea | verdict | status | pointer |
|---|---|---|---|
| Split-point sweep (one cut after block 1-12) | There is a cliff between blocks 6 and 7 (outlier 0.38 → 0.58). Block 7 is the earliest safe cut. | works (block 7 adopted) | log: "Split-point sweep"; records/online-token-router.md |
| First drop after block 1 / 3 / 5, recipe otherwise | 8-cell 0.839 / 0.872 / 0.880 vs 0.916 at 0.61 / 0.73 / 0.87x its FLOPs (fresh 3,904 rows): -0.02 to -0.04 at matched FLOPs. The block-7 cliff holds. | negative | log: "Recipe with its first drop after block 1" |
| Two-stage keep (20% @7, 7% @16) vs one stage | Best on the 1,464-row original pool: +0.11 outlier at matched FLOPs (2 seeds). Ties one stage on fresh pools and on the 11-task mix. | works (adopted) | log: "Two-stage keep"; ops/experiments/learned_router/STATUS.md |
| Stage 2 at block 12; stage-2 10%; 3 stages (+24:0.03) | Stage 2 at block 12 ties. Stage-2 10%: 0.25x; 3 stages: 0.23x. Both are close to the two-stage arm on per-task numbers (mt4, 2 seeds, learned_router STATUS). Stage-2 10% on mt11: 0.740. | neutral | log: "Router hill-climb round"; learned_router STATUS |
| Pyramid 50% @3 → 20% @7 → 7% @16 | 8-cell 0.870 vs 0.883 (1,952), 0.906 (3 seeds) vs 0.916 (3,904), 0.925 vs 0.924 (7,808) at 0.83x the recipe's FLOPs. At best ≈1.2x in FTM. | neutral | log: "Pyramid keep" |
| Aggressive keep trained directly (10%→3% / 10%→5%) | 8-cell 0.860 at 293 PF (FTM 3.1x) / 0.848 (2.4x) at 1,952; 0.909 (3.9x / 3.7x) at 3,904. Only 12-15% fewer FLOPs, because blocks 0-6 run dense (floor ≈0.2x/step). | neutral (not a 5x path) | log: "Aggressive keep trained directly" |
| Keep below 20%/7% [proxy] | Efficiency relative to the recipe: 10%/3.5% 0.71x, 5%/2% 0.55x, 2%/1% 0.27x (reported). | negative | [proxy] id=recipe; records/tokdrop-handoff-2026-11-17.md §2a |
| Keep fraction vs context length (2k-64k, does the best keep fall with length?) | Not yet scored. Session `keepfrac-len` on branch claude/keepfrac-vs-length-4bf10c, proxy metric. | in progress | records/tokdrop-handoff-2026-11-17.md §2a |
| Adaptive stage-2 budget (keep tokens covering 70% of predicted gradient mass, cap 50%) | Ties the recipe on fresh data (8-cell 0.880 / 0.918 / 0.922 vs 0.883 / 0.915 / 0.924). Loses 0.03 on the 11-task mix (xabsence -0.10, outlier -0.11). | negative | log: "Adaptive stage-2 budget"; log: "Hill-climb replicates" |
| Late keep-widening (35%/15% in the last third) at 7,808 rows | 2 seeds: f1 0.931 vs 0.926 (+0.005 ±≈0.003), CE -0.0035, for +8.6% FLOPs. Within seed noise. Seed 0's >4.1x was a lucky draw. | neutral | log: "Large-data decay"; branch claude/tokdrop-decay-fix |
| No full-context (dense) steps / routed uniform warmup instead of the dense step 1 | Dropping full-context steps is free. With a routed warmup (`--st-otr-warmup-random`) the recipe matches itself: 2-seed 0.848, 0.884 at 3,904, with peak memory 24.2 vs 26.9 GiB. | works | log: "Aggressive-cut screen"; records/weighted-token-sampling.md (wtsS0wr) |
| Position-id renumbering over the kept tokens after each cut | 0.820 vs 0.846 (1,952); 0.870 vs 0.882/0.889 (3,904). Almost all of the loss is on outlier. Keep the original positions. | negative | records/posid-reset-recipe-ablation.md (branch claude/posid-reset-0abac3) |
| Layer skip (fixed skips, LayerDrop 16-29, gradient-ranked layers), 44 runs | Dense + skips ties dense at 0.78-0.89x FLOPs. Recipe + skip loses 0.007-0.13 for 3-8% fewer FLOPs. Forward-skip selectors collapse. | negative | log: "Layer-skip screen"; records/layer-skip-screen.md (branch claude/train-layerskip-35010b) |
| Document-local attention in the early blocks [proxy] | Block 3 only: 0.01-0.02 held-out CE behind at every step for a 2.8% FLOP saving. More blocks stall training. Documents need cross-document attention in training. | negative | [proxy] id=lenscale; loop/lenscale |

## 3. LoRA, freezing and adapter scope

| idea | verdict | status | pointer |
|---|---|---|---|
| LoRA x routing 2x2 (each cell at its best LR) | Routing carries most of the gain. LoRA-all is 15-35% of it on the 4-task mix and shrinks with data on 11-task (54% → 19% → 7%). At matched data all arms are within ≈0.03: the gain comes from cheaper steps. | works (both kept) | log: "LoRA × routing 2×2"; records/lora-vs-routing-ablation.md (branch claude/vibrant-williams-4b0f66) |
| Dense LR and dense + LoRA-all controls | Dense is best at 5e-6 everywhere (1e-5 / 2e-5 worse). | reference | log: "Review controls" |
| Recipe LR / batch [proxy] | Efficiency relative to the recipe: LR 2e-5 0.85x, 4e-5 0.61x, 8 rows/step 0.74x. Keep LR 1e-5 and 16 rows/step. | negative | [proxy] id=hparams |
| LoRA r64 on blocks 0-6 only (frozen base) + router | Same quality as the plain router at 0.25x (vs 0.30x) and 48.4 vs 60.3 GiB peak (2 seeds). Rank 16 is worse. Superseded by whole-model LoRA. | works (superseded) | log: "LoRA early stack" |
| Freeze blocks 0-6 at 50% / 75%, progressive freeze | Fine on the 4-task mix (0.21-0.24x) but costs xabsence on the 11-task mix (0.54-0.58 vs 0.68-0.71). The recipe does not freeze. | negative | log: "Freeze variants"; learned_router STATUS |
| Freeze / no-grad early stack from step 0 | Loses contradiction (0.34-0.55 vs 0.66). | negative | log: "Aggressive-cut screen" |
| Frozen forward-only prefix (blocks 0-k, no backward) [proxy] | 0.149x/step, but held-out CE 0.282 at step 60 vs ≈0.19 for the recipe. Deeper freezing is worse. The early-block gradient carries most of the learning. | negative | [proxy] id=prefix; loop/prefix |
| Partial FFN width (75%) | -0.10. | negative | log: "Router hill-climb round" |
| FFN skip for context tokens in blocks 0-6 [proxy] | 0.193x/step, but held-out CE 0.274 vs 0.201 at step 50. Skipping every body-token FFN in blocks 0-3 stops learning. | negative | [proxy] id=ffnskip; loop/ffnskip |
| ReLoRA merge-restart (large-data decay fix) | No effect (CE +0.003 ±0.002 at 3,904). | neutral | log: "Large-data decay"; records/tokdrop-large-data-decay.md (branch claude/tokdrop-decay-fix) |
| LoRA on Qwen3-4B (full attention) | LoRA matters more there: routing alone gives 1.19x, and LoRA is the memory win (19.5 vs 57 GiB). | works | log: "Recipe on Qwen3-4B" |

## 4. Document-level drops

Trap: document-drop methods can look fine on the 4-task mean and collapse on **outlier**. The outlier generator
defines the answer as the uniquely smallest topic group, and random document dropping breaks that. Keeping gold
documents whole is a **training shortcut**: the kept set reveals the answer. Check per-task numbers, and outlier
first, before trusting any document-level rule.

| idea | verdict | status | pointer |
|---|---|---|---|
| Gold kept whole + trimmed context (gold-forced keep), any LR | Learns a shortcut: contradiction@8k 0.03-0.07 vs dense 0.89. Gold-blind rules at best tie dense at 0.57x FLOPs. In oracle training, keeping only the gold whole collapses too. | negative (trap) | log: "Fixed keep rules"; tokdrop_lr STATUS "Universal-rule screen" |
| Random non-gold document drop (gold kept, doc IDs kept) at recipe FLOPs | 8k 0.555 / 0.588; outlier → 0.00. With a group-aware outlier drop: 0.665 / 0.661 vs 0.846, FTM <1.08x. It scores 0.930 at 2k (above the recipe) but does not transfer to 8k. | negative | records/docdrop-nongold-baseline.md (branch claude/docdrop-nongold-42bb6b) |
| Gold-aware stochastic keep (gold prob raised, never 0/1), `grJ` [proxy] | 1.26x the recipe at matched loss on the main pool, but it stalls on outlier (held-out CE 0.169 vs 0.10-0.13) (reported). It does not stack with the 50/50 pool. Being retested at 32k/64k by `keepfrac-len`. | mixed | [proxy] id=goldrand; records/tokdrop-handoff-2026-11-17.md §2a |
| Position-preserving document subsampling of long rows (keep 25% of non-gold docs) [proxy] | 1.23-1.35x over the recipe on the 16k loss. Keeping 50%: 1.10-1.16x. Ties the 80/20 pool. One seed, 16k reading from 320 rows. | works (proxy only) | [proxy] id=longpos; branch loop/longpos (`--longpos-docsub-keep`, unmerged) |
| Per-row oracle + 1.15x random tokens | 8k 0.733 (outlier 0.408). Gold keep 0.774 vs distractor 0.334, and routed CE drops below dense (a leak). Random tokens do not fix a per-row leak. | negative | records/oracle-threshold-sweep.md |

## 5. Continued pre-training (CPT)

| idea | verdict | status | pointer |
|---|---|---|---|
| Token dropping in CPT (OLMo 3 7B, 32k-128k Longmino): router, input drops, slots, MoD, all-local attention, early-exit loss, partial MLP | Every variant loses to dense. | negative | log: "Token dropping in CPT"; records/olmo3-cptdrop-results.md |
| Whole-model LoRA r64 with frozen output layer, CPT (no dropping) | Never behind dense. FTM 4.8 / 4.3 / 3.2 / 2.1 / ≈1.4x at 48-960 steps (32k), at 0.68x FLOPs, peak 22.3 vs 35.2 GiB. At 64k / 128k: -0.0057 / -0.0046 loss at 120 steps. | works (CPT only) | log: "Whole-model LoRA with frozen output layer"; log: "Frozen-output-layer whole-model LoRA at 64k"; records/olmo3-cptdrop-results.md |
| Drop-CPT bases as a starting point for the router recipe [proxy] | 0.65-0.77x the recipe's efficiency, no better than the standard base. | negative | [proxy] id=cptbase |
| Drop-CPT bases fine-tuned in their own token-exit format [proxy] | Input-level exit degrades after 25-50 steps. Exit at block 8 ties the standard base (0.233 vs 0.238), and both trail the recipe. CPT bases start 0.25-0.32 CE worse. | negative | [proxy] id=cptnative |
| Matched SFT + CTC eval of the CPT checkpoints | Held for a decision. | held | log: "Matched SFT + CTC generation eval" |
| CPT gradient maps | The per chunk-pair gradient share falls ≈20x by 16k distance, then stays flat. A late-chunk oracle needs 20-31% of context tokens for cos 0.95. | diagnostic | records/cpt-gradient-maps.md |

## 6. Data mix, length and scale

Grader trap: on the 11-task mix and the proxy's validation suite, **reorder** at 8k/16k scores ~0 for every arm,
dense full fine-tune included. The cause is the strict permutation parser (an exact permutation of 1..n is required)
plus repetition loops. Exclude reorder 8k/16k from means; a tolerant grader exists but is not in the hub's 11-task
means.

| idea | verdict | status | pointer |
|---|---|---|---|
| Recipe at 64k on the 11-task mix (¼ epoch, 10x data) | +0.037 task average over dense at matched FLOPs, ahead on all 10 graded tasks, at 0.365x FLOPs and 24% less peak memory. Contradiction gain grows from +0.041 at 8k to +0.119 at 64k. 1 seed. | works | log: "Recipe at 64k on the 11-task mix"; records/mt11-64k-online-router.md |
| Online router (10% @7, one stage) at 64k, ¼ to 1 epoch | ≈+0.03 over dense at matched FLOPs at every budget (FTM ≈1.6-1.7x). Trails dense by ≈0.03 on the same data. | works (weaker than the recipe) | log: "64k study"; records/mt11-64k-online-router.md |
| Length generalization of 64k-trained models to 128k / 256k (YaRN x2) | The recipe gain persists: +0.046 / +0.033 / +0.054 at 64k / 128k / 256k, 6-task mean (⚠ eval_size 125 at 256k). | works | log: "11-task 64k: length-generalization evals"; records/mt11-64k-length-generalization.md |
| Train on rows <=16k (more rows, matched FLOPs) | About 0 at 2k-32k, then -0.05 to -0.10 at 64k-256k (mostly contradiction and hotpotqa). 2x the 16k data does not close it. | negative | same record (branch claude/mt11-lengen-bed337) |
| Train on rows <=32k (matched FLOPs) | Closes 51-77% of the 16k-vs-64k gap but stays 0.02-0.05 under 64k data, mostly on contradiction. Only 5.2% of rows exceed 16k. The <=32k recipe step costs 0.367x. | partial | log: "11-task 64k length generalization"; records/mt11-64k-length-generalization.md (branch claude/mt11-lengen-32k-56ed58) |
| 11-task mix at 8k with small data (480-720 rows/task) | The router failed strmatch (-0.12 to -0.25) and grouping. A per-doc floor of 16 fixes strmatch; with the floor the recipe leads dense by +0.04-0.05 (≈1.7x). | works (with floor) | mt11_64k STATUS "Question"; learned_router STATUS; records/online-token-router.md |
| Larger data (dense at 5,856 / 7,808, recipe at 7,808) | Dense saturates from 3,904 (0.928 / 0.931). The recipe's 0.926 at 7,808 equals dense at 3,904. On the old 8-cell metric FTM ≈2.0x (good only to 1.5-4x); the current 8k-only FTM is >=4.5x. The decay is mostly dense's ceiling plus seed noise. | works (ratio noisy) | log: "Scored dense at 5,856"; log: "Large-data decay" |
| Short-row mixes (80% 1-2k rows + 20% 4-16k) [proxy] | 3.1-3.7x vs dense LoRA over a full epoch (2 seeds), 3.2-3.5x on 16k rows (56 rows) vs 2.1-2.7x for the plain recipe (reported). At depth on the 320-row 16k set the advantage is gone: a 3x pool ≈ recipe at equal PF. | neutral at depth | [proxy] id=subproblem / rows; LOOP_HANDOFF.md |
| Rows <=5k only; length curriculum; token-budget batches [proxy] | Rows <=5k plateau at CE 0.119. A curriculum gives no gain over a static mix. Token-budget batches help dense LoRA (1.4-1.5x) more than the recipe. | neutral | [proxy] id=rows |
| Loss-proportional task / length sampling [proxy] | No gain at 16k (0.9-1.0x of the recipe on the same pool). | neutral | [proxy] id=alloc; loop/alloc |
| Multi-question rows (append 8-32 sub-questions) [proxy] | Same CE at more PF (0.124 at 220 PF vs 0.124 at 180-210). The sub-answers dilute the main answer's loss; a weighted variant was not tried. | negative | [proxy] id=multiq |
| Shared-context multi-query training; train only on the hardest 10-20% of rows | Not run (no drastic data changes, decided 10-29). | dropped | log: "Shared-context multi-query"; log: "Row selection" |
| Model scale, Qwen3.5 0.8B / 2B / 4B / 9B | The gain grows with size. FTM at 1,952: 1.73 / 2.42 / 4.08 / >4.42x (log; STATUS says 9B 4.8x). At 3,904: 1.88 / 3.22 / 3.68x. Routing, not LoRA, carries the 0.8B crossing. | works | log: "Model-scale ablation"; records/tokdrop-model-scale-ablation.md (branch claude/recipe-scale-ablation-71d007) |
| Qwen3 (all full attention) 0.6B / 1.7B / 4B / 8B | Generalizes with a smaller, flat gain: 2.9 / 2.9 / 2.1-3.3 / 2.3-2.6x (best split 12/24 on 4B). The accuracy loss at matched data is larger, mostly on outlier and contradiction. | works (smaller) | log: "Recipe on Qwen3-4B"; tokdrop_lr STATUS; branch claude/qwen3-recipe-c876d1 |
| Generation-eval validation of the proxy (16k pool, 3-task mean, reorder excluded) | The recipe needs 3.3x fewer FLOPs than dense LoRA (8k 3.5x), where the proxy said 1.9-2.2x. **Dense full fine-tune f1 0.928 beats the recipe f1 0.897 and dense LoRA 0.882.** Against full FT the recipe does not reach the reference. Use full FT as the stronger baseline. | caveat | tokdrop_lr STATUS "2026-11-17"; branch loop/validate @ 787581912 |

## 7. Memory

| idea | verdict | status | pointer |
|---|---|---|---|
| Max trainable row length at 80 GB (memory probes + four fixes) | Dense ≈165k, recipe ≈300k, whole-model LoRA ≈400k tokens per row. | works | log: "Max trainable length at 80 GB" |
| Chunked early blocks (4k) + 4,096 kept-token cap + host offload [proxy] | Memory slope falls from 0.37 GiB/1k tokens (dense LoRA) and 0.24 (recipe) to 0.040. A 1M-token row fits on one 141 GB GPU, and 2M fits at 126.5 GiB. Learning at 16k matches the recipe (⚠ eval_size 184). Learning at 64k+ and the keep cap's effect are untested. | works (memory only) | [proxy] id=memscale; flags `--prefix-chunk`, `--prefix-chunk-offload`, `--st-otr-keep-cap` (merged into claude/early-loss-proxy-d6e886) |
| Token-level vs 100/128-token chunk gather/scatter (microbenchmark) | Gather/scatter overhead is 1-5% of a layer at 32k, and chunking barely matters. At 8k, keeping 25% makes a GDN block only 30% faster. | neutral | log: "Per-layer skipping overhead" |
| Wall-clock vs FLOPs | Cheap methods run at 0.54-0.65x dense step time for 0.22-0.30x FLOPs. Predicted ≈0.29-0.47x after packing, fused LoRA and compile. | diagnostic | log: "Wall-clock vs FLOPs" |
| Sparse backward (row-sparse / sketched, exact forward) | 0.11-0.13x/step, but CE stalls at ≈0.3: 8-cell 0.51-0.69 vs 0.883. Unbiased sketches diverge. Kept from it: the answer-only LM head and the exact zero-row skip. | negative | log: "Backward compression"; records/sparse-backward.md (branch claude/sparse-backward) |
| Multiple views per row from a cached block-7 state | Every k>1 trails the recipe at matched FLOPs (k=2 -0.013 / -0.011; k=3 -0.069 / -0.017). Extra views act like extra epochs. | negative | log: "Multiple views per row"; records/multiview-cached-h7.md (branch claude/multiview-h7cache) |
| KV-only context for dropped tokens (to drop after block 1-3) | The fidelity probe closes only ≈10-35% of the gap to the block-7 split. No training run. | negative | log: "KV-only context"; branch claude/kv-only-context-9ee75b |
| Detached / reused context states (context forward-only) [proxy] | Context no-grad from block 0: 0.15x the recipe's efficiency. Every 4th step dense: 0.44x. | negative | [proxy] id=detach; loop/detach |
| Cheap stand-ins for dropped tokens (stale block-7 state, means, ridge corrections) | The best stand-in, a stale state fed to every layer, recovers expected grad cos 0.19-0.42 at 0.43x/step. None raises the per-step ceiling. ⚠ 128 rows/task (512 total). | negative | records/context-fidelity-probe.md (branch claude/gallant-robinson-adc54a) |

## 8. Mechanism and diagnostics (no new method; read before designing one)

| idea | verdict | status | pointer |
|---|---|---|---|
| Why does it work? Routed gradient on the trained recipe | On the recipe checkpoint the router-routed gradient is nearly unbiased (expected cos 0.87, norm 0.89). Random stays biased (0.18). On a dense checkpoint the router gives 0.04. The recipe works through co-adaptation, not gradient matching. | diagnostic | log: "Gradient sparsity vs routing gap"; records/grad-routing-gap.md |
| Early-phase gradient probe | The routed gradient is biased just after step 0 and unbiased by step ≈31 (≈6% of training). Aggressive budgets decay to ≈0.38. ⚠ eval_size 256/ckpt. | diagnostic | log: "Early-phase gradient probe"; records/grad-routing-gap.md |
| Whole-token oracle under true routing | Sets holding 95% of attribution give a routed cos of only 0.08-0.19. Attribution under the dense forward is not a proxy for routing. | diagnostic | records/oracle-tokens-under-routing.md |
| Gradient fidelity v1/v2 (bias vs variance; split points) | Early cuts fail by bias even for a perfect predictor (8-batch cos 0.41 / 0.08 / 0.08 / 0.03 for splits after 7 / 5 / 3 / 1). ⚠ 64 rows/task. | diagnostic | log: "Gradient fidelity v2"; log: "Gradient fidelity v1"; records/online-token-router.md |
| Router ranking accuracy | Spearman 0.51-0.67 vs its own target (⚠ 32 held-out rows/task). Its top 30% holds 73-95% of squared gradient. Block 7 vs 31 rank correlation is only 0.33-0.39. Never measured: the trained router vs the true gradient at blocks 16/24/31. | diagnostic | log: "Router ranking accuracy" |
| Gradient maps (token x layer) | Answer + query tokens (2-10%) carry 60-95% of blocks 7-31 energy. Gold docs carry 8-55x the per-token energy of distractors (nq / contradiction). Document gradient sits in blocks 0-15. ⚠ 2k maps: 128 rows. | diagnostic | log: "Gradient maps"; research/token_routing/grad_concentration/gradmaps/findings.md (branch claude/grad-support-tokens) |
| Gradient composition (SFT + CPT) | Blocks 0-6 hold 0.45 of the parameter gradient at the base and 0.20-0.40 after training. Rankings decorrelate with depth, which is the case for rescoring at block 16. | diagnostic | log: "Gradient composition probe" |
| Gradient concentration across training stages | Preliminary, 8 of 11 stages: at 366 steps block 16 needs 15% of tokens for 95% (the recipe keeps 7%). The log still lists it as running. | partial | log: "Gradient concentration across training stages"; branch claude/grad-concentration-d91a37 |
| Early dense-loss proxy itself [proxy] | Step-10/20 dense held-out CE ranks 19 arms with Spearman 0.83-0.84 against final 8k means (reported). It cannot resolve gaps under ≈0.03-0.05, and it underprices the recipe and misses full FT > LoRA. | caveat | LOOP_HANDOFF.md; records/early-dense-loss-proxy.md (branch claude/early-loss-proxy-d6e886) |
| Literature review (~70 papers) | No published long-context fine-tuning method reports more than ~1.4-2x at parity. Untested baselines: Random-LTD-style per-layer resampling, attention-score selection at block 7, an end-to-end MoD-style gate, shifted-sparse attention, and forward skipping of solved rows. | reference | records/lit-review-efficient-long-context-training.md |
