# LeakySFT: Qwen3.5-4B contradiction, one epoch

**Two-GPU retry after first-step optimizer-state OOM on one H100 80GB.** Per the 2026-11-18 data selection, all six arms use
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

All arms share a builder. Each requests **two GPUs / one replica on
ai2/jupiter-cirrascale-2, urgent, min-runtime=1h**. No context/tensor/pipeline
parallelism. Two-rank FSDP shards model/optimizer state and retains bf16 forward computation with fp32
reduction/master parameters; full activation checkpointing and fused linear CE
limit activation/logit memory. All one-GPU arms completed forward/backward but failed allocating FP32 Adam state.

Initial recipe: dense Qwen3.5-4B hybrid CPT
`amandab/qwen35-4b-dense-longmino/step2385/model_and_optim`, strict weights-only
loading, fresh optimizer, LR1e-5, AdamW betas(0.9,0.95), weight decay zero,
linear decay with 3% warmup, seed34521. Source: only contradiction, no Dolci or
CPT text, no subsampling. BFD whole-document packing into 32768-token windows.
Two windows per optimizer step (65,536 global tokens), one per GPU, and one loader
epoch: all 4,986 measured packed windows are consumed in 2,493 steps, with no
incomplete-batch tail. Prep: `01M7QMF46AHRKA5344ZEHMFZDN` (2026-11-18).
This doubles the batch from the failed single-GPU attempts; optimizer precision,
LR and data exposure are unchanged. Rendered examples longer than the window are
excluded, never truncated; preparation must quantify these exclusions on the
selected dataset before launch.

All six arms have identical data order and loss-validity exclusions. Only the
context-loss scaler changes. Explicit zero uses the LeakySFT masking/normalization
and diagnostic path, rather than the legacy SFT code path. Compute and gradient
magnitudes need not match across scalers. This is a new two-GPU pilot, not a
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
fingerprints, packed windows, and one-epoch steps. Prep retained 17,630 examples (162,506,388 tokens), excluded 2,358 overlong
examples, and packed 4,986 windows. If submitted through Gantry,
this CPU preparation job must also request `--min-runtime 1h`.
Validate the two-GPU retry through optimizer steps before relaunching all arms. The frozen core tests' nine
CUDA cases have not run on this local CPU environment.

Validation: `src/test/scripts/leaky_sft_sweep_test.py` checks the exact six
scalers, identical normalized configs, one epoch, strict initialization, model
construction, resource flags, selected source and the exact 32K filter boundary. The original frozen
LeakySFT tests remain unchanged.

## Regular-SFT throughput control

`Qwen3.5-4B-contradiction-32k-SFT-speed.py` uses `cpt_loss_scaler=None`,
selecting the original SFT path without context-loss diagnostics. It hard-stops
at 300 steps while retaining the original one-epoch LR schedule, two GPUs,
65,536-token global batch, checkpoint cadence, seed, and input stream.
Compare steps 51–300 with the same steps in the LeakySFT jobs; report checkpoint
intervals separately from steady-state throughput. Launch with the same Gantry
resources (Jupiter, urgent, minimum runtime 1h) and a fresh run/save name.

## Implementation audit (2026-11-19)

The completed sweep used commit `1866c6113` and the token-weighted mean
`(answer_sum + alpha * context_sum) / (answer_count + alpha * context_count)`.
The original frozen tests encode this objective. It is **not** an interpolation
between mean SFT and mean CPT loss. On this corpus there are 522,027 answer
and 161,966,731 valid context targets (310.27 context targets per answer target).
At alpha=0.1, context has 96.88% of aggregate token weight, and its per-token
coefficient is 97.19% of full CPT, rather than 10%. The last recorded alpha=0.1
batch attributed 99.986% of its numerical loss to context. Changing the objective
requires an explicit new experiment contract; the audit does not silently change it.

Alpha=0 also differs from legacy SFT's distributed normalization: LeakySFT averages
by global target count, whereas legacy SFT averages rank-local means. In 2,394 of
2,493 actual batches the two ranks have unequal answer-target counts. Every answer
label/mask in the packed dataset nevertheless matches the legacy label builder.

The audit additionally reproduced and fixed:

- The shared fused CE wrapper passed a nonzero z-loss coefficient even with
  `compute_z_loss=False`. Liger 0.6.4's boolean controls returning the auxiliary
  loss, not enabling it. This affected both the historical regular fused SFT and
  LeakySFT paths. Explicit zero z-loss coefficients were also replaced by a default.
- Biased fused heads crashed on alpha=0 context diagnostics under `no_grad`, due
  to a Liger 0.6.4 missing bias-gradient dereference. A bounded CE fallback handles
  these diagnostic-only calls. The Qwen3.5 sweep uses a bias-free head.
- Non-fused FP16 loss sums could overflow on long contexts; empty-group zero
  anchors could overflow as well. These paths now avoid low-precision reductions.
  The sweep used fused BF16 loss, so these FP16 issues do not explain its scores.

`audit_test.py` adds reproductions without modifying the three frozen test files.
`audit_distributed.py` checks actual FSDP gradients against explicit full-batch CE.
`audit_data.py` measures the real packed corpus and retrieves recorded diagnostics.

## NormalizedLeakySFT

The corrected method is a separate opt-in on `TransformerTrainModuleConfig`:

```python
normalized_leaky_sft_alpha=0.1,
cpt_loss_scaler=None,
```

For each DP rank's **whole optimizer batch**, define `S_out`, `S_ctx`, `N_out`,
`N_ctx` over the existing valid target masks. The objective is

```
(1 - alpha) * S_out / N_out + alpha * (S_out + S_ctx) / (N_out + N_ctx)
```

DP averages these rank-local objectives, just like regular SFT. Microbatches use
these same batch denominators, so accumulation does not reweight short answers.
Thus alpha=0 matches legacy SFT normalization for identical valid answer labels;
alpha=1 is the all-valid-token CPT mean with the same rank averaging. At alpha=.1,
each context target has exactly .1 of its coefficient in that CPT objective,
regardless of the answer/context ratio. This describes objective coefficients,
not a guarantee that context contributes 10% of the numerical loss or gradients.
Empty groups contribute differentiable zero; denominators are clamped to one.
For instance filtering, both means retain legacy SFT's convention of counting
excluded instances as a full sequence of zero-loss slots.

Implementation reuses the tested disjoint fused CE sums, with output coefficient
`(1-alpha)/N_out + alpha/N_all` and context coefficient `alpha/N_all`. It passes
their ratio and reciprocal output coefficient to the existing head. All masking,
packing, and structural-token exclusions remain unchanged. The old
`cpt_loss_scaler` objective remains available for historical reproducibility;
setting both methods raises an error. Diagnostics use `train/normalized_leaky/*`.
The existing TP/PP and z-loss restrictions also apply to the new method.

The shared fixes for unintended fused z-loss, biased-head no-grad diagnostics,
and FP16 overflow apply to both methods. `normalized_test.py` was committed before
implementation; the original three LeakySFT test files are still unchanged.
`AUDIT_NORMALIZED=1` enables a two-GPU fused-loss gradient audit with unequal
answer counts and two microbatches per rank, including direct alpha=0 comparison
to the regular SFT code path. No new sweep is launched by this implementation.

### NormalizedLeakySFT sweep (2026-11-19)

| Method | Arms | Control |
| --- | --- | --- |
| NormalizedLeakySFT | a00 / a01 / a03 / a05 / a07 | Same Qwen3.5-4B base and one-epoch filtered contradiction corpus as the original sweep |

Launch with `launch_sweep.py --normalized --prefix NEW_UNIQUE_PREFIX --ref PUSHED_SHA --output-dir NEW_DIR --submit`.
The five named NormalizedLeakySFT entry points share the original builder. Only
objective selection and W&B group differ in scientific configs; data order,
4,986 packed windows, 2,493 steps, batch size, optimizer, and checkpoint policy are
unchanged. The prior prep is reused because neither data nor packing changed.
All launches use the audited Liger 0.6.4 and the shared kernel/numerical fixes.
Two GPUs per run, Jupiter, urgent, min-runtime=1h. No alpha=1 arm is included.
