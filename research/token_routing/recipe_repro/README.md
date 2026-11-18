# Reproducing the FLOP-drop recipe on Beaker

Start here. This page shows how to reproduce the recipe's dense-vs-recipe Pareto curves on three data mixes, how to
read the results, and how to run your own variants. Everything runs on Beaker with weka; nothing needs Berkeley
machines.

- **What else was tried, and how it turned out:** [`WHAT_WAS_TRIED.md`](WHAT_WAS_TRIED.md).
- **State of this hand-off (verified vs not yet run, job ids):** [`HANDOFF.md`](HANDOFF.md).

Tags used below:

- **(not yet run)**: the command or number has not been checked by an actual run on this branch.
- **⚠ eval_size N**: the number comes from fewer than 500 eval examples.

`n` always means documents per row (corpus size). Eval-set size is always written `eval_size`.

## 1. The recipe in one paragraph

We fine-tune Qwen3.5-4B (36 blocks: Gated DeltaNet plus full attention) on synthetic long-context tasks from the CTC
suite. The recipe has two parts:

- **Whole-model LoRA:** rank 64, scale 32, on all 32 adapted blocks, LR 1e-5.
- **Online token router:** a ridge regression, refit every step on per-token gradient norms from the training backward
  pass. Every token runs blocks 0-6. After block 7, each document keeps its top 20% of body tokens; after block 16, it
  keeps the top 7%. Dropped tokens leave the stack. Prompt, document markers, query and answer are always kept, and
  kept tokens keep their original position ids.

Inference is dense over the full context. The arm name is `lora64all32r20to7`.

| per training step (4-task 8k mix, 16 rows/step) | dense | recipe |
|---|---|---|
| training FLOPs (FLOP meter) | 1.0x | 0.22x (345 vs 1,531 PF per 488 steps) |
| peak GPU memory | not recorded here | 26.9 GiB (reported in the 11-17 hand-off) |
| wall-clock on 2 H100s, 488 steps | 1.53 h | 0.72 h |

**FLOPs-to-match (FTM)** = how many times more training FLOPs dense needs to reach the recipe's score. It is read off
the dense curve, interpolated linearly in log FLOPs.

## 2. Results to reproduce

Every number below is printed by `python research/token_routing/recipe_repro/repro.py reference --mix <mix>`. The
numbers come from `reference.json`, which `build_reference.py` built from Prasann's results log and the training logs
on Beaker. All of them passed `ops result check`. The only warnings were the reorder cells and 2k cells at exactly 1.0.

- Scores are f1, except oolong (`score`) and grouping (`pairwise_f1`).
- eval_size is 500 per task and rung, no-cot.
- **PF** = petaFLOPs of training, the last `flop_meter/actual_pflops` value in the run's log.
- The ± is the binomial SE of the mean, from eval rows only. Seed-to-seed spread on a 4-task 8k mean is about
  ±0.01-0.02 on top of it.
- Rank by the **8k rung** first.

### 2a. `fast8k`: the established result

4 tasks (contradiction, nq, outlier, oolong), rows up to 12,288 tokens. Budgets are in fresh rows per task. One seed
per run, except the recipe at 1,952 rows/task, which has 2 seeds.

| arm | rows/task | PF | 8k mean | 2k mean | FTM at 8k |
|---|---|---|---|---|---|
| dense | 488 | 379 | 0.667 | 0.805 | |
| dense | 976 | 762 | 0.712 | 0.816 | |
| dense | 1,952 | 1,531 | 0.843 | 0.933 | |
| dense | 3,904 | 3,063 | 0.899 | 0.953 | |
| dense | 5,856 | 4,595 | 0.901 | 0.955 | |
| dense | 7,808 | 6,128 | 0.904 | 0.958 | |
| **recipe** (seed 0 / 1) | 1,952 | 345 / 343 | **0.845 / 0.847** | 0.921 / 0.919 | **4.6x / 4.7x** |
| **recipe** | 3,904 | 688 | **0.882** | 0.949 | **3.6x** |
| **recipe** | 7,808 | 1,373 | **0.904** | 0.944 | **≥4.5x** |

SE on each 8k mean is ±0.006-0.009. "≥" means the recipe beats every dense point, so FTM is a lower bound.

Dense saturates from about 3,904 rows/task, so FTM at large data is poorly determined (anywhere from 1.5x to 4x).
On 2k the recipe trails dense at matched data: this is why 8k is the primary metric.

### 2b. `new4`: the newer 4-task mix (less established)

4 tasks (contradiction, outlier, reorder, qdmatch_nq), rows at 4k, 8k and about 14.5k tokens, 1,952 rows/task (one
epoch = 488 steps). This is a single validation sweep from 2026-11-17: one seed, except the recipe at f1, which has 2.

**Reorder is excluded from every mean.** Its grader is a strict-permutation parser, and it scores every arm, including
dense full fine-tuning, at about 0 (parse rate about 0.65). Means are therefore over 3 tasks.

The FTM reference here is dense LoRA (rank 64 on every block, LR 1e-5), the only dense curve with three points. Dense
full fine-tuning was run at one budget only, and it beats the recipe.

| arm | budget (epoch) | PF | 8k mean | 16k mean | all-rung | FTM at 8k vs dense LoRA |
|---|---|---|---|---|---|---|
| dense LoRA | 1/4 | 374 | 0.760 | 0.615 | 0.743 | |
| dense LoRA | 1/2 | 754 | 0.847 | 0.702 | 0.825 | |
| dense LoRA | 1 | 1,496 | 0.906 | 0.772 | 0.882 | |
| dense full FT | 1 | 1,982 | **0.946** | **0.850** | **0.928** | (one point only) |
| **recipe** | 1/2 | 223 | 0.850 | 0.696 | 0.822 | **3.5x** |
| **recipe** (seed 0 / 1) | 1 | 442 | 0.931 / 0.925 | 0.792 / 0.794 | 0.899 / 0.895 | **≥3.4x** |

SE on each 8k mean is ±0.006-0.010.

Most other claims about this mix come from the paused early-loss proxy. Those are held-out answer loss on 184 rows
(⚠ eval_size 184), mostly one seed, so treat them as rankings, not prices. Details are in `WHAT_WAS_TRIED.md`.

### 2c. `mt11`: the 11-task mix at 64k

11 tasks, rows up to 65,536 tokens, 128 rows/step. Budgets are fractions of one epoch (1,223 steps). Ten tasks are
scored; rerank has no grader. One seed, LR 5e-6 for both arms.

The recipe on this mix is the earlier variant `lora64r20to7f16`:

- LoRA only on blocks 0-6.
- At least 16 kept tokens per document.

Whole-model LoRA was never run at 64k.

| arm | budget (epoch) | PF | 8k mean (9 tasks) | 64k mean (6 tasks) | all-rung (42 cells) | FTM at 8k |
|---|---|---|---|---|---|---|
| dense | 1/16 | 1,175 | 0.758 | 0.460 | 0.668 | |
| dense | 1/8 | 2,287 | 0.813 | 0.518 | 0.714 | |
| dense | 1/4 | 4,625 | 0.846 | 0.583 | 0.761 | |
| dense | 1/2 | 9,294 | 0.881 | 0.638 | 0.804 | |
| dense | 1 | 18,592 | 0.905 | 0.708 | 0.848 | |
| **recipe** | 1/4 | 1,696 | **0.831** | **0.538** | **0.732** | **2.0x** |

SE on each 8k mean is ±0.004-0.006.

Excluded as no-signal: reorder at 8k and 16k (every arm fails the parser), and grouping at 32k (every arm scores
0.001-0.005).

The record `records/mt11-64k-online-router.md` uses a different summary that gives the same picture: the recipe is
+0.037 per task on average against the per-cell better of dense LR 2e-6 and 5e-6, ahead on 10/10 tasks. The gain holds
past the training length, to 128k and 256k (`records/mt11-64k-length-generalization.md`; ⚠ eval_size 125 at 256k).

## 3. Setup on Beaker (as a different user)

### 3a. Code

```bash
git clone https://github.com/allenai/OLMo-core.git && cd OLMo-core
git checkout prasann/flopdrop_clean
git checkout -b <you>/recipe-spinoff          # your own branch; never commit to prasann/flopdrop_clean
git push -u origin <you>/recipe-spinoff
pip install -e '.[beaker]'     # launchers only (beaker-py + gantry); training runs in the Beaker image
```

Start from `prasann/flopdrop_clean` and do all your work on your own branch cut from it.

The launchers submit from your laptop or any machine that has the `beaker` CLI configured. Gantry builds the job from
the pushed commit, so push your branch before every launch (`git push`).

The job image is `OLMoCoreBeakerImage.stable`. The training code installs itself from the repo inside the job.

### 3b. Your own settings (env vars; no file edits)

```bash
export BEAKER_WORKSPACE=ai2/<your-workspace>   # default: ai2/flex2 (Prasann's)
export BEAKER_BUDGET=ai2/<your-budget>         # default: ai2/oe-other
export WANDB_ENTITY=<your-wandb-entity>        # default: Prasann's entity
export WANDB_PROJECT=<your-project>            # default: memory-networks
export CTC_USER_DIR=<your-weka-folder>         # checkpoints + eval outputs go to
                                               #   /weka/oe-training-default/ai2-llm/checkpoints/$CTC_USER_DIR/ctc_suite/ckpts/<run>
export RR_PREFIX=<your-initials>               # run-name prefix; keeps names unique
```

Your Beaker workspace needs the secrets `<BEAKER_USER>_BEAKER_TOKEN` and `<BEAKER_USER>_WANDB_API_KEY`. These are the
names that `build_launch_config` in `src/olmo_core/internal/common.py` looks up.

Jobs request GPUs on `ai2/jupiter-cirrascale-2` (H100). Use `--cluster` to choose another cluster, or a comma list to
take whichever cluster has room first.

### 3c. Inputs on weka (read-only, shared)

All inputs live under `W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns`.

| what | path under `$W` |
|---|---|
| base model: Qwen3.5-4B with repaired marker embeddings | `ctc_suite/bases/q35-4b-base-markerfix/model_and_optim` |
| tokenizer (same vocabulary as the 4B) | `hf_tokenizers/Qwen3.5-0.8B-Base` |
| `fast8k` training shards (`<rows>` = 488 … 7808) | `ds64/fast8k/shards/mt4f<rows>` |
| `fast8k` eval files | `outlier_lengthmix/eval_rungs/{outlier,nq}/`, `_eval_bundle_eval500_v3/contra/`, `_eval_bundle_eval500_v2_clean/oolong/` |
| `new4` training pool | `early_loss_proxy/shards/mtb16k` (description: `../early_loss_proxy/POOL_V2.md`) |
| `mt11` training pool | `mt11_64k/shards/mt11L64` |
| `new4` / `mt11` eval rows (CTC-suite rows, 500 per rung) | `_eval_bundle/ctc_suite_rows/<subset>/rung_<tokens>.jsonl` |

**Why the base is "markerfix":** Qwen3 never trained the reserved marker tokens that delimit documents. Their
embeddings are identical, so the model cannot tell "open document" from "close document", and runs train to chance in
a way that looks like a modeling result. Always start from the repaired base. For a new base, run
`recipes/data/fix_marker_embeddings.py` first.

**Access:** all of these paths exist, and Beaker jobs that mount weka `oe-training-default` can read them, because
those jobs run as `root`. This was checked on 2026-11-17 by `check_access_beaker.sh` (Beaker job
01M7NZ6PJ6W5PFRXXTGT0SGB98, launched from Prasann's account; every path printed `OK`).

- Data and eval files are world-readable.
- The base checkpoint's files are mode `600` (owner `root`). Read them only from inside a Beaker job, not from a
  non-root shell on a weka-mounted machine.

To repeat the check from your own account, run `bash research/token_routing/recipe_repro/check_access_beaker.sh`. It
launches a CPU-only job of a few minutes.

## 4. Launch

Every training run is one Beaker job, launched through `recipes/train/research_trainer/beaker_ctc_suite.py`. Each
`repro.py` command below sets the job up as follows:

- priority `urgent`
- minRuntime = min(4 h, 1.3 x the expected runtime), and at least 1 h; never 0
- for a run expected to take over 4 h, a rolling resume point (`--resume-interval`, about every 30 min) so a
  preemption does not restart from step 0
- the run name and Beaker experiment id are saved to `runs_state.json` in this folder (git-ignored)

**Before every real launch, add `--dry`.** It prints the resolved Beaker config and submits nothing.

### 4a. Train: one command per point on the curve

```bash
R=research/token_routing/recipe_repro/repro.py

# fast8k: dense at 6 budgets, the recipe at 3 (2 GPUs each)
for b in f488 f976 f1952 f3904 f5856 f7808; do python $R train --mix fast8k --arm dense --budget $b; done
for b in f1952 f3904 f7808;                 do python $R train --mix fast8k --arm recipe --budget $b; done
python $R train --mix fast8k --arm recipe --budget f1952 --seed 1          # second seed

# new4: dense LoRA curve, the recipe, one dense full-FT point
for b in f4 f2 f1; do python $R train --mix new4 --arm denselora --budget $b; done
for b in f2 f1;    do python $R train --mix new4 --arm recipe --budget $b; done
python $R train --mix new4 --arm dense --budget f1                         # 4 GPUs

# mt11: dense at 5 budgets (8 GPUs), the recipe at 1/4 epoch (4 GPUs). Expensive: see section 5.
for b in f16 f8 f4 f2 f1; do python $R train --mix mt11 --arm dense --budget $b; done
python $R train --mix mt11 --arm recipe --budget f4
```

**Budget names:**

| mix | budget name | meaning |
|---|---|---|
| `fast8k` | `f<rows>` | the first `<rows>` fresh rows per task, one epoch |
| `new4`, `mt11` | `f<k>` | 1/k of an epoch; uses `--max-tokens`, so the LR schedule (3% warmup, linear decay) completes |

Every arm trains unpacked, 16 rows/step (`mt11`: 128), with micro-batch 2 (`mt11`: 1). FLOPs do not depend on the GPU
count, so `--gpus 4` is safe when 2-GPU slots are scarce.

**Verified so far:**

- The `fast8k` recipe command ran end to end on Beaker at the smallest budget (f488, 2026-11-17): 15 min on 2 H100s,
  86.5 PF, keep fractions 0.205 / 0.079, peak memory 27 GiB, and the export was written. Its evals, and the dense
  f488 run, are **(not yet run to completion; see HANDOFF.md)**.
- Every command above passes as a `--dry` config (all seven mix x arm combinations, and the eval commands).

### 4b. Eval

```bash
python $R eval --mix fast8k --run $RR_PREFIX-fast8k-recipe-f1952   # one 2-GPU job per task; omit --run = every run of the mix
python $R status                                                  # Beaker state of everything in runs_state.json
```

All three mixes use the same evaluator: the repo's native long-context ladder
(`recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py`, docchunk variant). It does chunked prefill over the
document markers, then greedy decoding. Settings: 500 rows per rung, no-cot, tokenizer `Qwen3.5-0.8B-Base`.

| mix | rungs | eval rows | query position | max length |
|---|---|---|---|---|
| `fast8k` | 8k, 2k | the 2026-09 ds64 eval files (table in 3c) | after | 20,000 |
| `new4` | 4k, 8k, 16k | CTC-suite rows | after | 40,960 |
| `mt11` | 2k, 8k, 16k, 32k, 64k | CTC-suite rows, where the subset has them | both | 73,728 |

The reference numbers were all scored this way, so use this evaluator, not olmo-eval, to compare against them.

The olmo-eval suite launcher only matters past 64k. It lives on olmo-eval branch `prasann/ctc-suite-flopdrop`, script
`scripts/ctc_suite/launch_ctc_suite.py`, tokenizer `Qwen/Qwen3.5-0.8B`. At r256k it applies YaRN x2 with chunked
prefill. The `mt11` 128k/256k numbers come from it; see `records/ctc-fast-suite-eval.md`. It agrees with the native
ladder at 64k to within ±0.002.

### 4c. Turn results into the curve and FLOPs-to-match

```bash
python $R collect --mix fast8k      # reads the Beaker logs: PF per run, score per task x rung
python $R reference --mix fast8k    # the same table for Prasann's runs, to compare against
```

`collect` prints one row per run. Each row has the PF, the 8k mean (or the mix's primary rung), the other rung means
and the all-rung mean, each with its SE and cell count. For every non-dense run it also prints the FTM against your
own dense curve. It saves the table to `results_<mix>.json`.

The FTM rule (function `ftm()` in `repro.py`):

- The dense curve is the running maximum of score vs PF, interpolated linearly in log PF.
- A method that scores below the cheapest dense point gets `<=` (an upper bound).
- A method that scores above the most expensive dense point gets `>=` (a lower bound).

## 5. Cost per run and what a healthy run looks like

### 5a. Measured cost per run

These are the reference runs, on H100s (jupiter).

| mix | arm | budget | GPUs | wall-clock | GPU-h |
|---|---|---|---|---|---|
| `fast8k` | dense | f1952 / f3904 | 2 | 1.5 h / 3.0 h | 3 / 6 |
| `fast8k` | dense | f5856 / f7808 | 4 | 2.3 h / 3.0 h | 9 / 12 |
| `fast8k` | recipe | f1952 / f3904 / f7808 | 2 | 0.7 h / 1.4 h / 2.8 h | 1.4 / 2.8 / 5.5 |
| `new4` | dense LoRA | f4 / f2 / f1 | 2 | 0.6 h / 1.0 h / 2.0 h | 1.1 / 2.1 / 4.1 |
| `new4` | recipe | f2 / f1 | 2 | 0.4 h / 0.8 h | 0.9 / 1.6 |
| `new4` | dense full FT | f1 | 4 | 1.0 h | 4.0 |
| `mt11` | dense | f16 / f8 / f4 | 8 | 0.6 h / 1.1 h / 1.9 h | 5 / 9 / 16 |
| `mt11` | dense | f2 / f1 | 4 | 6.4 h / 13.6 h | 26 / 54 |
| `mt11` | recipe | f4 | 4 | 3.1 h | 12 |

Not included in the table:

- **Queue time.** It is often longer than the run itself.
- **Evals.** Each eval is one 2-GPU job per (run, task): 4 jobs per `fast8k` / `new4` run and 10 per `mt11` run.

Full reproduction, training only: roughly 45 GPU-h for `fast8k`, 15 for `new4`, and 120 for `mt11`.

### 5b. What a correct run looks like

| check | where to look | healthy value |
|---|---|---|
| FLOP meter | training log, `flop_meter/actual_pflops` | `fast8k` f1952: about 345 (recipe) vs 1,531 (dense); `new4` f1: about 442 vs 1,496 (dense LoRA) |
| router keep fractions | log lines `[online-router] step N: keep_frac=… s1_keep_frac_abs=…` | about 0.20-0.21 kept after block 7; about 0.07-0.08 of the original after block 16 (per-document rounding up adds a little) |
| training loss | wandb / log `train/CE loss` | falls from about 1.0 to about 0.15 within the first few steps |
| final training loss | `fast8k` f1952 | about 0.07-0.08, last step, noisy |
| eval sanity | eval log lines `[ladder:<task>@<rung>] f1=… (n=500, …)` | n=500 per rung |

- **An 8k mean far below the table is a bug.** One cause: a `--max-length` below the prompt truncates it away, and the
  run scores exactly 0.000.
- **A recipe run that costs about the same PF as dense** means the router flags did not reach the trainer.

## 6. Running your own spin-off

### 6a. Where the code lives

| what | path |
|---|---|
| router: scoring, ridge refit, per-document allocation, mid-stack cuts | `src/olmo_core/nn/online_token_router.py` |
| router callback (logging, freezing) | `src/olmo_core/train/callbacks/online_router.py` |
| trainer and every `--st-*` flag | `recipes/train/research_trainer/train_ctc_suite.py` (see `--help`) |
| Beaker wrapper | `recipes/train/research_trainer/beaker_ctc_suite.py` |
| arm vocabulary of past experiments (about 100 named arms) | `research/token_routing/ds64_fast8k/launch_fast8k.py` (`ARM_EXTRA`) and `research/token_routing/online_router/launch_otr.py` (`ARMS`) |

### 6b. The flags that matter

| flag | recipe value | what it controls |
|---|---|---|
| `--st-otr-stages` | `7:0.2,16:0.07` | after block L, keep fraction f of each document's body tokens; stages are nested |
| `--st-otr-doc-share` | `1.0` | 1 = budget per document (the only allocation that survived); 0 = budget per row |
| `--st-otr-score` | `router` | how tokens are ranked: `random`, `tail`, `first_last`, `oracle` (dense backward; costs 2-3x per step) |
| `--st-otr-explore-frac` | `0.1` | share of the kept budget given to random tokens, which supply router training targets |
| `--st-otr-dense-every` | `0` | N > 0 = every Nth step trains on full context (charged in full) |
| `--st-otr-early-lora`, `--st-otr-early-lora-scale`, `--st-otr-early-lora-blocks` | `64`, `32`, `32` | LoRA rank, scale, and how many blocks get adapters (32 = whole model) |
| `--st-otr-min-per-doc` | `0` | at least K kept tokens per document |
| `--st-otr-reset-positions` | off | renumber positions after each cut (this hurts: −0.026 at 8k) |
| `--max-tokens` / `--max-steps` | | budget with a complete LR schedule / hard stop mid-schedule |
| `--resume-interval` | | rolling resume point every N steps |

### 6c. Add an arm

Give a new `--arm` name and its full flag string. A new arm defaults to the soft-token path at LR 1e-5; set `--lr` to
change that.

```bash
# a new arm: the recipe with stage 2 at 10% instead of 7%
python $R train --mix fast8k --arm r20to10 --budget f1952 --dry --flags \
  "--st-keep-token-rule online --st-drop-slots --st-keep-prob 0.0 --st-otr-keep-frac 0.2 --st-otr-explore-frac 0.1 \
   --st-otr-doc-share 1.0 --st-otr-dense-every 0 --st-otr-drop-layer 7 --st-otr-stages 7:0.2,16:0.10 \
   --st-otr-early-lora 64 --st-otr-early-lora-scale 32 --st-otr-early-lora-blocks 32"

# or extra flags on top of a named arm
python $R train --mix fast8k --arm recipe --budget f1952 --tag floor16 --flags "--st-otr-min-per-doc 16" --dry
```

To compare against the reference without rerunning dense, put the new arm's PF and 8k mean on the dense curve
(section 2a). `collect` prints its FTM once your dense points exist.

The project's working rule: an idea needs a **best-case per-step FLOP saving** and an **empirical FTM**, and
**≥5x** over dense is the bar for a new approach.

### 6d. The cheap screen, and its limits

The cheapest real screen is `fast8k` at f1952 (about 1.4 GPU-h for a recipe-like arm, plus 4 eval jobs). Compare it
with the recipe's 0.845 / 0.847 (2 seeds).

- One seed resolves only about ±0.02 on the 8k mean. A promising arm needs a second seed, and a point at f3904.
- A cheaper proxy exists: the dense held-out loss after 10-50 steps, on branch `claude/early-loss-proxy-d6e886`,
  folder `research/token_routing/early_loss_proxy/`. It ranks arms well (Spearman 0.83-0.95 against final 8k means),
  but it cannot resolve gaps under about 0.03-0.05. It also understated the recipe's FLOP gain: 1.9-2.2x, against 3.3x
  from generation evals. Use it to reject ideas, not to price them.

### 6e. Traps that each cost a day

- **Reusing a run name silently resumes the old run.** The save folder is keyed by the run name, and the trainer
  auto-resumes from any checkpoint it finds there. Use `--tag` for every rerun; `repro.py` refuses a name it has
  already launched.
- **Read FLOPs from `flop_meter/actual_pflops` of a finished run.** Do not use the total/dense-equivalent number or a
  number from a run that is still going.
- **`--max-tokens` below one step's worth of tokens** gives a zero-step or one-step run that still "succeeds". Budget
  with whole steps (`steps x rows/step x seq_len`).
- **`F8K`-style sequence length below the longest row silently skips rows.** It must be at least the shard's
  `max_example_len`: 12,288 for `fast8k`, 16,384 for `new4`.
- **The mean hides per-task collapse.** Document-drop methods look fine on the 4-task mean and fall to 0.0-0.4 on
  outlier. Always read the per-task table and the worst task.
- **Keeping gold documents whole is a training shortcut.** "Which document is complete?" then names the answer. The
  model learns it during training (contradiction 0.07 vs 0.89), and the shortcut does not exist at eval. Keep
  selection gold-blind, or keep matched decoys.
- **Exactly 0.000 or 1.000 is a bug signature until checked.** Exceptions: reorder (broken grader) and some 2k cells
  at ceiling.
- **Evaluating a LoRA checkpoint needs this branch's eval code.** The eval code rebuilds the adapters; older eval
  commits silently score the plain base.
- **Marker-embedding repair** (section 3c) is needed for any new base model.
- **Urgent Beaker jobs are preempted soon after minRuntime.** Never set minRuntime to 0: those jobs wait for days.
  For runs over 4 h, keep the resume point.

## 7. Optional: Prasann's Berkeley tooling

None of this is needed on Beaker.

- `ops` (`/accounts/projects/berkeleynlp/prasann/bin/ops`) is Prasann's run registry and results log.
  - The launchers append a registry row through `ops/opslib` (in this repo) and print a warning instead if that
    fails. Either way the launch goes ahead.
  - `build_reference.py` reads it to rebuild `reference.json`.
- Paths under `/data/...`, `mooney:`, `sneetches:` and slurm partitions (`jsteinhardt`, `berkeleynlp`) in records and
  older scripts are Berkeley-only.
- The GPU smoke runs listed in `HANDOFF.md` ran on Berkeley slurm.

## 8. Further reading

- `WHAT_WAS_TRIED.md`: every idea tried and its verdict.
- `records/tokdrop-handoff-2026-11-09.md`: goal, recipe and mechanism.
- `records/tokdrop-handoff-2026-11-17.md`: latest state.
- `records/online-token-router.md`: router design and the hill-climb results.
- `records/mt11-64k-online-router.md` and `records/mt11-64k-length-generalization.md`: the 64k mix.
- `../early_loss_proxy/POOL_V2.md`: how the `new4` pools and held-out sets were built.
- `../early_loss_proxy/validation/`: the `new4` reference runs (`results.json`, `table.json`, collector).
- `research/token_routing/ds64_fast8k/build_mt4_fresh_beaker.sh`: how the `fast8k` pools were built.
- `research/token_routing/mt11_64k/build_beaker.sh`: how the `mt11` pool was built.
