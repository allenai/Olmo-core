# LeakySFT: Qwen3.5-4B contradiction, one epoch

**Prepared, not submitted.** Per the 2026-11-18 data selection, all six arms use
`prasanns/xlong5_2k256k_qwen35/shards_full/contradiction_train`, retaining only
complete examples whose tokenized length is **<=32,768**. Length includes the
rendered prompt, answer and separator. Exactly-32K examples are retained;
longer examples are excluded before BFD packing, never truncated or split.
One epoch means one pass over this retained subset, not the full 256K corpus.

| Arm | cpt_loss_scaler | Role |
| --- | ---: | --- |
| a00 | 0.0 | Answer-only SFT control, context CE diagnostic |
| a01 | 0.1 | LeakySFT |
| a03 | 0.3 | LeakySFT |
| a05 | 0.5 | LeakySFT |
| a07 | 0.7 | LeakySFT |
| a10 | 1.0 | All-token language-model objective on the same SFT corpus |

All arms share a builder. Each requests **one GPU / one replica on
ai2/jupiter-cirrascale-2, urgent, min-runtime=1h**. No context/tensor/pipeline
parallelism. Single-process FSDP retains bf16 forward computation with fp32
reduction/master parameters; full activation checkpointing and fused linear CE
limit activation/logit memory. Actual one-GPU memory fit remains unvalidated.

Initial recipe: dense Qwen3.5-4B hybrid CPT
`amandab/qwen35-4b-dense-longmino/step2385/model_and_optim`, strict weights-only
loading, fresh optimizer, LR1e-5, AdamW betas(0.9,0.95), weight decay zero,
linear decay with 3% warmup, seed34521. Source: only contradiction, no Dolci or
CPT text, no subsampling. BFD whole-document packing into 32768-token windows.
One window per optimizer step and one loader epoch: all packed windows are
consumed, with no incomplete-batch tail. Steps are the measured packed-window
count, not a guessed token budget. Rendered examples longer than the window are
excluded, never truncated; preparation must quantify these exclusions on the
selected dataset before launch.

All six arms have identical data order and loss-validity exclusions. Only the
context-loss scaler changes. Explicit zero uses the LeakySFT masking/normalization
and diagnostic path, rather than the legacy SFT code path. Compute and gradient
magnitudes need not match across scalers. This is a new single-GPU pilot, not a
reproduction of the previous 16-GPU contradiction recipe.

## Usage

Inspect a configuration locally:

```bash
PYTHONPATH=src python src/scripts/train/memexpress/leaky_sft/Qwen3.5-4B-contradiction-32k-1ep-LeakySFT-a00.py dry_run RUN_NAME ai2/jupiter-cirrascale-2
```

Generate the six launch commands without submitting anything:

```bash
python src/scripts/train/memexpress/leaky_sft/launch_sweep.py --output-dir /tmp/leakysft-plan
```

Use `--ref FULL_PUSHED_SHA` once the implementation, tests and all launcher files
are committed and pushed. The optional `--submit` executes the commands detached
and appends resource settings, source SHA, status and submission-log paths to
`launch-ledger.jsonl`. A new output directory and timestamped run names avoid
accidental repeat submissions/resumes. The legacy `launch` subcommand is disabled
because its BeakerLaunchConfig lacks a minimum-runtime field.

Before submitting, run the CPU preparation on a Weka host:

```bash
PYTHONPATH=src python src/scripts/train/memexpress/leaky_sft/prep.py --output /path/to/new-prep-report.json
```

It checks source metadata and checkpoint presence, builds the real filtered
packing plan, and records retained/excluded examples and tokens, source
fingerprints, packed windows, and one-epoch steps. These counts have not been
measured locally because Weka is not mounted here. If submitted through Gantry,
this CPU preparation job must also request `--min-runtime 1h`.
Validate CUDA/Liger plus one-GPU fit before the sweep. The frozen core tests' nine
CUDA cases have not run on this local CPU environment.

Validation: `src/test/scripts/leaky_sft_sweep_test.py` checks the exact six
scalers, identical normalized configs, one epoch, strict initialization, model
construction, resource flags, selected source and the exact 32K filter boundary. The original frozen
LeakySFT tests remain unchanged.
