---
title: Random non-gold document dropping at the recipe's FLOPs (gold kept, IDs preserved) — a document-level baseline for the token-routing recipe
date: 2026-11-13
updated: 2026-11-13
experiment: tokdrop_lr
area: token_routing
kind: result
status: negative (11-13): far below the recipe at equal FLOPs (best arm 8k mean 0.665 vs 0.846, single seed); the plain drop breaks outlier's label (0.00); contradiction loses 0.16-0.26 at 8k
superseded_by: 
artifact: 
summary: Drop a random 84% of each row's non-gold documents before block 0 (gold kept, printed IDs unchanged, oolong whole) at the recipe's training FLOPs (0.225x dense, 350 PF at 1,952 rows/task), whole-model LoRA r64 at LR 1e-5 as the recipe. 8k 4-task mean 0.555-0.588 with the plain drop (outlier collapses to 0.00 because random dropping makes the outlier topic the only multi-passage group, the opposite of its definition) and 0.665 with a group-aware outlier drop, vs recipe 0.846 and dense 0.843. FTM < 1.1x (below the lowest dense point).
---
# Random non-gold document dropping (`docdrop`)

Prasann (2026-11-12): "drop random x% of non-gold documents (match recipe flops) but preserve document IDs for full
contexts. Handle any edge-case task-specific behavior however you want." Code: `research/token_routing/docdrop/`
(branch `claude/docdrop-nongold-42bb6b`). Eval unchanged: dense, full context.

## Verdict

**No.** At the recipe's training FLOPs, random document dropping with gold kept is far below the recipe on the
primary metric: the best arm reaches an 8k 4-task mean of **0.665** (group-aware outlier drop, original positions)
against the recipe's 0.846 / 0.847 and dense 0.843 (single seed each; seed noise on the 8k mean is about 0.02-0.03).
It is below dense f488 (0.667 at 379 PF), so its FTM is an upper bound, **< 1.1x**, against the recipe's 4.6x.
Best-case per-step FLOP speedup is the same as the recipe's (4.4x vs 4.5x), by construction. Where it loses:

1. **outlier** (plain drop 0.00; group-aware 0.42 vs recipe 0.86). The plain drop is not label-preserving, see below.
2. **contradiction** (0.70-0.80 vs recipe 0.96). Labels stay valid; the model trained on 34-claim contexts with a
   6x higher gold share and does not transfer to 187-claim contexts. No over-prediction (3.0 pairs per answer,
   exactly the gold count); the pairs it names are wrong more often (precision 0.70-0.80).
3. **nq** a little (0.886-0.896 vs 0.912); **oolong** (kept whole) 0.63-0.66 vs 0.650, within noise.

The pattern across rungs is the clearest finding: with a valid outlier label the doc-drop model is at or above the
recipe at the **2k** rung (`ddpg` 2k mean 0.930 vs 0.921; outlier 0.988) and far below it at **8k**. Its training rows
are ~1.3-1.8k tokens with 8-34 documents and 3 topic groups, which is what the 2k rung looks like; it has never seen
an 8k context of 48-187 documents and 14 topic groups at training time. The recipe trains on the full 8k contexts
(blocks 0-6 see every token) and keeps its 8k score. Dropping whole documents trades away exactly the long-context
exposure the 8k metric measures.

Per the brief (clearly below, so no seeds and no 3,904-rows point).

## Setup

- Data: the fresh 4-task pool `mt4f1952` (1,952 rows/task, mooney `/data/prasann/oracle/shards/mt4f1952`). Every arm
  trains 488 steps of 16 rows, one epoch, the same rows in the same order (seed 0) as the recipe.
- Optimisation = the dense-LoRA arm `f8k-mt4-denselora64all-*` flag for flag (`--variant full --lora-rank 64
  --lora-blocks 0 --lora-scale 32`: whole-model LoRA r64, scale 32, embeddings frozen, LM head trained, the
  recipe's adapter set) at the recipe's LR 1e-5, 2 H200, micro 2, `--seq-len 12288`. The only change is the input.
- Drop (`build_docdrop_shard.py`): a row is prompt + documents + query/answer; each document is one
  `<|box_start|>...<|box_end|>` span that prints its own ID ("Claim 37: ...", "Document [37]: ..."). **Gold = the
  IDs the answer span cites, matched to the IDs the documents print** (no sidecar, so no index-base question; every
  printed ID equalled its position + 1 and every cited ID was found, 0 issues over 5,856 rows). Keep every gold
  document and `round(q * n_nongold)` uniformly random non-gold ones (seeded per row, nested in q), cut the rest out
  of the row. Kept documents are byte-identical, so their IDs are the full-context IDs; the answer is unchanged.
- `q` = 0.157 (drop 84% of non-gold documents), set by bisection on the trainer's per-example FLOP formula so the
  run costs what the recipe costs. The formula reproduces the dense f1952 run (1,528.7 PF modelled vs 1,531 measured).
- Measured with the FLOP meter (`flop_meter/actual_pflops`): 350.6 PF (plain), 352.5 PF (group-aware), vs recipe
  343-345 and dense 1,531. Step cost 0.229 / 0.230 of the dense run (0.225 / 0.226 of the meter's own
  dense-equivalent on the same rows; the recipe's 0.221 is the latter kind). Both position arms charge exactly the
  same PF, a check that they kept the same documents.
- FLOP meter fix in this branch: with `--lora-rank` and per-example accounting the frozen-base / adapter credit is
  now priced at each example's own length, not at `--seq-len` (which credited a 1.5k-token row with 12k-token
  attention it never ran). Unit test in `src/test/train/callbacks/flop_meter_layer_drop_test.py`.

### Realised keep shares (q = 0.157, both drop variants)

| task | docs per row | gold per row | docs kept | tokens kept | gold share of visible docs, before -> after | row length after (mean / max) |
|---|---|---|---|---|---|---|
| contradiction | 187 | 6 (3 pairs) | 18.2% (34) | 19.7% | 3.2% -> 17.6% | 1,649 / 2,373 |
| nq | 48 | 1 | 16.7% (8) | 17.3% | 2.1% -> 12.5% | 1,325 / 1,725 |
| outlier (plain) | 57 | 3 | 19.3% (11) | 20.5% | 5.3% -> 27.3% | 1,775 / 2,226 |
| outlier (group-aware) | 57 | 3 | 19.9% (11.3) | 21.1% | 5.3% -> 26.6% | 1,823 / 2,415 |
| oolong | 30-200 items | - | 100% | 100% | - | 4,263 / 8,813 |
| all | | | 34.2% | 31.2% | | |

oolong is 14.7% of the pool's tokens and is charged in full, which is why the other tasks keep only ~17-21%.
Non-document tokens (instructions, query, answer) are 64-144 per row on the three dropped tasks; no row comes near
the budget on them alone (the budget is global, not per row).

## Decisions

- **Where: the data path for compact positions, the soft-token compaction for original positions.** `ddc` trains on a
  new shard whose rows simply lack the dropped documents (transparent: every row can be decoded and checked);
  `ddp` trains on the full rows and drops the same documents before block 0 with the existing soft-token compaction
  (`--st-gold-sidecar keep_sidecar.json --st-n-random 0 --st-drop-slots`; the sidecar maps each full row's content
  fingerprint to its kept documents; 100% fingerprint hits). Dropped documents cost no FLOPs in either.
- **Positions: both arms run.** `ddc` = compact (what a short context naturally has); `ddp` = original full-context
  positions with gaps (what the recipe uses after compaction).
- **oolong whole.** Its label aggregates over every item and the per-item labels are not in the tokens (nor exactly
  recoverable from the generator without the source rows); its prompt also states the item count. Leaving it whole
  and charging its FLOPs is exact; q for the other tasks rises to hold the budget.
- **contradiction, nq: plain random drop.** Labels stay valid: a pair contradicts whatever else is present, fillers
  are random claims from the same corpus (no grouping), and nq's single relevant document stays relevant. Neither
  prompt states a document count.
- **outlier: the plain drop is invalid; the fix arm drops whole topic groups.** See below.

## outlier: random dropping inverts the task

The outlier generator (`ctc/src/ctc/tasks/outlier/generate.py`) builds each row from whole Wikipedia articles: 13
majority articles of >= 4 passages each and one outlier article of exactly 3 passages. "The outlier" is the uniquely
SMALLEST topic group; the generator's own warning is that dropping random distractors can leave a majority topic
smaller than the outlier. Mapping every training passage back to its article through the seed pool
(`outlier_groups.py`, 1,952 rows, 0 unmatched) confirms it: min majority group 4 in every row, outlier group 3.

Keeping 8 random non-gold passages of 54 leaves the majority articles at 0-2 passages each, while the 3 outlier
passages always survive together. In the training context the outliers are then the **only** multi-passage topic,
the opposite of the task's definition, and the label is no longer supported by the kept context. The trained model
learned "name the passages that share an article": on the 8k eval, 99.2% of `ddp`'s answers name passages from one
single majority article (`outlier_eval_groups.py`, 490 rows aligned by gold), 0.1% of its predicted IDs are true
outliers (recipe on the same rows: 84%). That is below the ~5% chance rate, with well-formed answers (3.0-3.6
IDs, no empty or unparseable generations).

**Fix arm (`ddpg` / `ddcg`): group-aware drop on outlier.** Keep the outlier article and whole majority articles in a
random order, as many as bring the kept non-gold count closest to q (always 2 articles, 8-10 passages, at this
budget). Every visible majority topic stays larger than the outlier topic (smallest kept 4-5 vs 3; asserted per
row), so the label is valid and "passages that share a topic" is no longer a tell. contradiction and nq unchanged.
One row (112) has two near-identical "Timeline of Orthodoxy in Greece" articles whose passage keys collide; they are
merged into one group.

The gold-share shortcut the brief anticipated ("whatever is present is the answer") did not show up as
over-prediction on any task: answers name exactly the gold count. The outlier failure is a label-validity problem,
and the contradiction loss is a difficulty / length transfer problem.

## Results (1,952 rows/task, eval_size 500 per cell, no-cot, single seed)

SE per cell 0.01-0.02 (outlier at 0.4: 0.022); SE of the 4-task mean about 0.008-0.01.

| arm | positions | outlier drop | PF | step cost (x dense) | best-case speedup / step | peak GiB/GPU | contra 8k | nq 8k | outlier 8k | oolong 8k | **8k mean** | 2k mean | FTM (8k) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense (full FT, LR 5e-6) | - | - | 1,531 | 1.0 | 1.0 | - | 0.973 | 0.924 | 0.817 | 0.656 | 0.843 | - | 1.0 |
| dense LoRA r64 (full rows, LR 1e-5) | - | - | ~1,128 (model) | 0.74 | 1.36 | - | 0.974 | 0.922 | 0.858 | 0.659 | 0.853 | 0.914 | - |
| recipe `lora64all32r20to7` (s0 / sd1) | original | - | 343-345 | 0.221 | 4.5 | 26.9 | .960 / .956 | .912 / .922 | .860 / .861 | .650 / .649 | **.846 / .847** | .921 / .919 | 4.6x |
| `ddp` | original | random docs | 350.6 | 0.229 | 4.4 | 25.6 | 0.696 | 0.886 | 0.001 | 0.635 | **0.555** | 0.672 | < 1.08x |
| `ddc` | compact | random docs | 350.6 | 0.229 | 4.4 | 26.7* | 0.799 | 0.890 | 0.002 | 0.662 | **0.588** | 0.684 | < 1.08x |
| `ddpg` | original | whole groups | 352.5 | 0.230 | 4.3 | 25.6 | 0.715 | 0.896 | 0.421 | 0.627 | **0.665** | **0.930** | < 1.08x |
| `ddcg` | compact | whole groups | DDCGPF | | | | DDCG | | | | | | |

2k per task: `ddp` 0.868 / 0.962 / 0.000 / 0.856; `ddc` 0.898 / 0.978 / 0.002 / 0.858; `ddpg` 0.902 / 0.976 / 0.988 / 0.854 (2k mean 0.930, above the recipe's 0.921).
References at 1,952 rows (8k mean): random token keep after block 7 at 19.7% 0.841 and at 11.7% 0.796; content-blind
tail rule 0.781. FTM: every docdrop arm is below dense f488 (0.667 at 379 PF), so FTM = 379 / PF is an upper bound.
Step cost = PF / dense f1952's 1,531 PF; speedup = 1 / step cost.
\* `ddc` trains padded rows (the dense path pads every row to `--seq-len` 12,288), so its memory is the full-length
row's; the compacting `ddp` arms show what the method needs (25.6 GiB, set by the longest row, a whole oolong row).

Compact vs original positions: compact is better on contradiction (0.799 vs 0.696 at 8k, ~4 SE) and level elsewhere
(single seed).

## Limits

- Single seed per arm (stopped per the brief: clearly below). No 3,904-rows point.
- The group-aware outlier drop always keeps exactly 2 majority articles; a model can still learn "the smaller of
  three groups", which transfers only partly to 14 groups (0.42).
- oolong stays whole by choice; a relabelled oolong would let the other tasks keep more.

## Where things are

- Code: `research/token_routing/docdrop/` (README lists files); unit tests `docdrop_test.py` (CPU) and the meter test.
- Shards: mooney `/data/prasann/docdrop/shards/mt4f1952_dd` (plain) and `mt4f1952_ddg` (group-aware), each with
  `docdrop_stats.json`, `docdrop_examples.txt`, `keep_sidecar.json`; outlier article map
  `/data/prasann/docdrop/outlier_groups_f1952.json`; eval group analysis `/data/prasann/docdrop/outlier_eval_groups_8k.json`.
- Runs: mooney `/data/prasann/docdrop/runs/f8k-mt4-{ddc,ddp,ddcg,ddpg}-f1952-dd-lr1e-05-loc` (flops.json, model);
  evals `/data/prasann/f8k_eval/runs/<run>/`; scores `ops result show --train-run <run>` (experiment `tokdrop_lr`).
