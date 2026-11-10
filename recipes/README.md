# recipes/ — the canonical CTC pipeline: data → train → eval

This page is the reproducible path behind the setA results: dense Qwen3.5-4B SFT on **setA**, the
11-task CTC mix built IID with the olmo-eval CTC suite at every rung from 2k to 256k, scored on the
suite. Experiment variants live in `research/<family>/`; this folder is the part that should always
work end to end.

| Step | Where | Record |
|---|---|---|
| 0. env | `env/local_env.sh` (local slurm), `pip install -e '.[all]'` + `pip install -e ctc/` | `local_cluster.md`, `beaker.md` |
| 1. data | `data/ctc_sft/build_ctc_sft.py` → `data/convert_unified_to_document_landmark.py` | `records/ctc-fast-suite-eval.md` |
| 1b. base repair | `data/fix_marker_embeddings.py` | `records/document-chunked-marker-embeddings.md` |
| 2. train | `train/setA/launch_setA_sft.py` | `train/setA/README.md` |
| 3. eval | olmo-eval `scripts/ctc_suite/launch_ctc_suite.py` (other repo) | `records/ctc-fast-suite-eval.md` |
| audits | `eval/setA_audit/`, `eval/ctc_eval_speed/` | `records/ctc-fast-suite-eval.md` |

Data, checkpoints and eval outputs live outside git (weka `.../prasanns/ctc_sft_sets/`,
`.../prasanns/<run-name>/`, `.../<beaker user>/_olmoeval_ctc/`).

## 1. Data

Task rows come from the vendored `ctc/` package (`ctc-data`; see `ctc/VENDORED.md`), except two
tasks built by in-repo generators in `data/generators/` (xabsence: one-sided Gutenberg exact-copy;
grouping: unlabeled OpenAlex). `data/ctc_sft/build_ctc_sft.py` declares every task once (`SETS`),
builds each (task, bucket), records every omission in `manifest.json`, and with `--convert` tokenizes
to shards with `convert_unified_to_document_landmark.py` (chat template, query position `both`, no
CoT, each document wrapped in `<|box_start|>`/`<|box_end|>`).

```bash
pip install -e ctc/
python recipes/data/ctc_sft/build_ctc_sft.py --set set-a \
    --out /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets \
    --buckets 2k 4k 8k 16k 32k 64k 128k 256k \
    --calibration recipes/data/ctc_sft/eval_calibration.json \
    --external-pools <dir with pool_gutenberg_train.jsonl + openalex_compact.jsonl> \
    --convert --seq-len 262144 --shards-dir shards_qwen35_256k
# -> <out>/setA_max20_evaliid/shards_qwen35_256k/<task>/   (--check re-hashes against the manifest)
```

The flags above are reconstructed from the record; the set's own `manifest.json` holds the exact
arguments of the shipped build. `--calibration` sizes every rung from the eval's own rows
(`eval_calibration.json` is made by `data/ctc_sft/measure_eval_ladder.py`) and builds into
`<tag>_evaliid`. The IID audit against real eval rows is `eval/setA_audit/audit_train_eval_iid.py`
(76/77 cells clean on the shipped set).

**Base checkpoint.** Qwen never trains the marker rows. Before training from a new base, audit and
repair it with `data/fix_marker_embeddings.py` (`fix_marker_embeddings_gantry.sh` on Beaker), and
train from the repaired copy. `prep` (below) re-runs the audit read-only.

## 2. Train

Two arms of one recipe (amandab's validated dense 256k SFT): `32k` (examples longer than 32,768
tokens dropped, not truncated) and `256k` (everything). Both: 1,048,576 tokens/step, LR 4e-5, one
epoch, one 8×H100 node, urgent on jupiter (allocated via oe-scaling), pinned to a pushed commit.

```bash
L=recipes/train/setA/launch_setA_sft.py
python $L prep            # CPU pre-flight: shards, drops per window, marker audit, packed steps
python $L 32k
python $L 256k            # add --dry-run to print the gantry command
```

**Single-task debugging and other inputs.** `--tasks nq` (comma list; default all 11),
`--data-root <set root>` and `--base-checkpoint <model_and_optim>` are exported to the job as
`SETA_TASKS` / `SETA_DATA_ROOT` / `SETA_BASE_CKPT` and read by `_qwen35_setA_common.py` (the
training scripts run under the olmo-core CLI, so they take env vars rather than flags). A
non-default task set is added to the run name:

```bash
python $L prep --tasks nq && python $L 32k --tasks nq     # -> q35-4b-dense-ctc-setA-nq-32k-<stamp>
```

**Research trainer.** `train/research_trainer/train_ctc_suite.py` is the task-agnostic torchrun
trainer that the research arms drive (`--variant full|chunked|chunked-mix|sparselandmark|pooledkv|
ffnmoe|softtoken|kvroute|flexcompute`), with local (`run_ctc_local.sbatch`), Lambda (`run_ctc_lambda.sbatch`) and Beaker
(`beaker_ctc_suite.py`) launchers and HF → olmo-core base conversion. See its README. The setA SFT
above does not use it.

**Variants, one place.** What each research arm changes relative to dense (model config,
`enable_*` calls, router LR groups, callbacks, data overrides, eval-side export) lives in
`train/variants.py` as a registry of `Variant` objects with typed option dataclasses. Both trainers
call it: `train_ctc_suite.py` builds the options from its argparse flags (the option fields are the
argparse dest names), and the setA recipe takes `launch_setA_sft.py --variant <name>
--variant-opts k=v,...` (see `train/setA/README.md`, "Research variants on the canonical recipe").
`train/check_variant_parity.py` is the CPU-only proof that the two agree and that the refactor
changed neither trainer's config; run it on a compute node:

```bash
# from a fresh `git archive` of HEAD (new) and of the pre-refactor commit 633919f00 (old) on /data
PYTHONPATH=<new>/src python <new>/recipes/train/check_variant_parity.py --old-root <old> --new-root <new>
```

## 3. Eval

The current path is olmo-eval's CTC launcher (`~/projects/olmo-eval`, branch
`prasann/ctc-suite-launcher`, `scripts/ctc_suite/README.md`): one single-GPU Beaker job per packed
bin, olmo-eval's native `olmo_core` provider, chat-format prompts IID with the SFT data.

```bash
cd ~/projects/olmo-eval && git checkout prasann/ctc-suite-launcher && uv sync
uv run python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm full \
    --doc-markers --rows setA --run-name <name> --dry-run      # plan, then drop --dry-run
uv run python scripts/ctc_suite/launch_ctc_suite.py --run-name <name> --collect    # table + eval_size + SE
uv run python scripts/ctc_suite/launch_ctc_suite.py --run-name <name> --resubmit   # preempted jobs only
```

- **`--arm full --doc-markers`** for setA checkpoints: the markers make each eval prompt an exact
  token prefix of its training sequence. Without the flag the prompts are out of format.
- **Tokenizer:** keep the default `Qwen/Qwen3.5-0.8B` (same vocabulary as the 4B). The 4B template
  opens `<think>\n` for a generation prompt, a split training never saw; `preflight.py` refuses it.
- **r256k runs with YaRN x2 + 131,072-token chunked prefill by default.** Its prompts reach ~272k
  tokens; in the base window 43-100% of rows were left-truncated (a fake 256k collapse). Quote
  r256k numbers as "YaRN x2". `--fit-window` (shorten distractors) is the alternative and agreed
  within noise on the setA pair.
- Eval sizes default to 300 (≤32k) / 100 (≥64k) per rung — below the 500 floor, so carry each
  cell's `eval_size` and SE (`--collect` prints both).
- `--olmo-core-ref` must carry the FLA autotune fix (default `prasann/landmark`).

**Dev-loss evals** (teacher-forced CE on the gold answer, used to screen routing/keep schemes
cheaply) live in `research/token_routing/devloss_grid/`; see `records/devloss-grid-underperformers.md`.

**Legacy native path (legacy).** `eval/legacy_native/` holds the in-repo native driver
(`ctc_eval/eval/eval_lc_native.py`), the multirung Beaker launcher
(`run_q4b_beaker_multirung_eval.py` → `run_beaker_multirung_eval.sh`), its local sbatch twin, and
`ctc_eval/launch_ctc_evals.py`. It scores the older singletask/xlong eval bundles (v2/v3 ladders,
`records/v3-eval-howto.md`) and is what the `run-evals` skill drives for those. Put
`recipes/eval/legacy_native` (not `src/scripts`) on `PYTHONPATH` to import `ctc_eval`.
`eval/make_yarn_copy.py` writes the config-only YaRN serving copies it needs past the trained context.
