# Mixed midtraining

Run `src/scripts/train/Mixed-Midtraining.py` after the joint phase of
[vision alignment](vision_alignment.md), using the standard internal experiment CLI.
The training recipe is `olmo_core.internal.vision_midtraining`; sources and loss allocation
are defined in `olmo_core.internal.vision_midtraining_data`. Configuration uses ordinary
dataclasses and dotted overrides. Data paths and launch defaults require access to Ai2
infrastructure.

## Configure and launch

Inspect a configuration without loading weights, allocating text sources or replaying data:

```bash
python src/scripts/train/Mixed-Midtraining.py dry_run mixed-midtraining local \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN
```

`recipe.text_loss_share` accepts values in `(0, 1]`. The default, `0.9`, allocates 90% of
expected supervised-loss weight to text and 10% to vision. Set it to `1.0` for text-only
training (T100):

```bash
python src/scripts/train/Mixed-Midtraining.py dry_run text-midtraining local \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN \
  --recipe.text_loss_share=1.0
```

T100 does not construct visual sources or forward dummy image crops. It retains the text
allocation, model, LM/connector optimizer settings and schedule, but freezes vision with
zero vision LR and no vision activation checkpointing. In mixed runs, an all-text local
batch forwards dummy crops to keep vision collectives aligned across ranks.

Launch with the standard CLI:

```bash
python src/scripts/train/Mixed-Midtraining.py launch mixed-midtraining ai2/holmes \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN
```

Defaults are two eight-GPU nodes, EP8, urgent priority, an eight-hour minimum runtime,
32 GiB shared memory and workspace `ai2/molmofication`. Change these with `--launch.*`
overrides. The launcher clones a remote commit; `allow_dirty` does not upload local or
untracked changes, so unpublished code requires an explicitly frozen source deployment.
`prep` is an optional dataset-preparation command; `train` runs within an existing
distributed allocation.
Inspect `launch.env_secrets` and configure credentials for the submitting account and
workspace. Defaults reference `JASONR_BEAKER_TOKEN`, `RUSTINS_WANDB_API_KEY` and
`GOOGLE_CREDENTIALS`; the GCS text recipe does not require AWS credential mounts.

The launcher sets `NVSHMEM_REMOTE_TRANSPORT=none` for node-local EP8 groups. Initialization
checks same-host membership and full GPU peer access before starting NVSHMEM; cross-node
NCCL data parallelism is unaffected. Cross-node EP requires a remote NVSHMEM transport
and compatible network interfaces instead. Direct `torchrun` users must set the transport
environment explicitly.

## Training defaults

| Setting | Default |
| --- | --- |
| Parent | Completed joint-alignment checkpoint; model-only handoff |
| Trainable components | Vision encoder, connector and LM, including full input/output embeddings |
| Expected supervised-loss allocation | 90% text / 10% vision |
| Context / global sequences / microbatch per GPU | 8,192 / 128 / 2 |
| Gradient accumulation | Four microbatches per GPU on 16 GPUs |
| Token-position budget | 50B, rounded up to 50,000,297,984 (47,684 steps) |
| Learning rates | LM `1e-5`; connector `2e-5`; vision `1e-6` (zero vision weight decay) |
| Schedule | 200-step warmup, cosine decay to 10% at the token budget |
| Activation checkpointing | LM blocks, vision encoder and connector |
| Packing | 48-example buffer, 64-crop packed-sequence limit, eight loader workers |
| Image preprocessing | Eight local detail crops plus one overview per image |
| Checkpoints | Every 10k steps, keeping two; latest temporary checkpoint every 500 steps |

The model architecture, tokenizer and router coefficients come from the alignment parent.
No router-input RMS repair or weight rescaling is applied. Execution uses FlexAttention,
EP8 dispatch and block recomputation. The default MB2/pack64 means two sequences per GPU
microbatch and at most 64 crops per packed sequence; the per-image crop limit is separate.
Recheck throughput and memory when changing the architecture, context, microbatch or image
preprocessing.

For a frozen-vision control, use the ordinary component overrides:

```bash
--train_module.freeze_params='["vision.*"]' \
--train_module.vision_activation_checkpointing=false
```

Frozen parameters are excluded from the optimizer.

`recipe.max_tokens` controls both the text allocation and scheduler horizon. The budget is
independent of `recipe.text_loss_share`; shares control sampling, not dataset construction.
Use `trainer.hard_stop` or a shorter `trainer.max_duration` to end a run earlier; neither
rescales the learning-rate schedule. Change `recipe.max_tokens` when changing both the
allocation and schedule horizon.

Scheduled positions include image tokens and padding. For comparisons, match scheduled
positions and report realized text/vision exposure and applied updates/skips separately;
equal position budgets do not imply equal text exposure.

## Checkpoint handoff and resume

Start from a completed joint-alignment checkpoint with compatible model and tokenizer
metadata. A fresh handoff loads model weights and resets optimizer, loader and trainer state.
Resuming the unchanged recipe in its existing output folder restores full state, including
packing and RNG state. Output folders must be separate from the alignment parent.
Full-state resumes require compatible configuration and runtime; preserve deployments
referenced by serialized config/callback classes.

## Sources and allocation

Text uses `OLMo3-32B-midtraining-modelnamefilter.yaml`, the 61-source mixture including its
instruction sources, with Dolma2 tokenization and repetition filtering. An explicit
`recipe.text_dataset` supports another compatible fixed-length text dataset and is required
for a different parent tokenizer.

The eight visual sources have the following default conditional example allocation:

| Sources | Share of visual example draws |
| --- | ---: |
| PixMo captions/transcripts, Points Basic, Points High Frequency, grounded Count and CoSyn Point | 70% |
| Scalar PixMo Count | 10% |
| OCR/document: TextVQA, DocVQA, InfoVQA and ChartQA | 12% |
| Audited alignment: filtered VisualWebInstruct and Geo170K | 8% |

The five-source core group retains its configured example ratios. Captions/transcripts share
one adapter; grounded and scalar counting are sampled independently. Grounded prompts request
coordinates, while scalar questions use integer targets. Basic/high-frequency points include
both pointing and point-count styles for each annotation. Caption and Basic-point training
reuse alignment selections. OCR and audited alignment retain prepared train/validation
exclusions and quality filters. The visual sources do not include Tulu.

`recipe.visual_example_weights` controls relative example draws within vision, independently
of response length. `recipe.visual_loss_shares` is empty by default; it optionally reserves
conditional loss shares for sources removed from the example-weight mapping. The remaining
visual mass preserves the example ratios. Sampling uses `target_loss_mass / mean_loss_weight`.
Loss shares are not input-token, image-count or gradient-norm shares; example shares are not
crop/compute shares or exact per-update quotas.

CE is normalized by the globally summed loss-mask weight. Visual response tokens are
unweighted, and text/vision sequence quotas are not fixed.
Filtered native-text windows retain their denominator weight, matching pretraining; inspect
both `data/source/*/loss_weight` and `active_loss_weight` when checking realized delivery.
Router auxiliary losses use real input-token masks and exclude padding independently of CE.

Sources are ordinary configs in `dataset.sources`. To add or replace a source, supply its
config, calibrated `dataset.mean_loss_weight.SOURCE`, and its membership in one of the two
visual weight mappings. The groups must be disjoint and cover the visual sources. Final loss
targets and training diagnostics are derived from these settings after component overrides.
Use the text-share knob rather than overriding `dataset.target_loss_mass` directly.

### Calibration

[visual_calibration_v1.json](../../../configs/vision_moe/vision_midtraining/visual_calibration_v1.json)
records the default means: 128 examples per source, seed 6198, pinned Dolma2 tokenizer,
8K context, eight local crops, document formatting, explicit grounding prompts and unweighted
response loss. These are bounded estimates, not exact corpus means. Recalibrate changed
populations, tokenizer or serialization; do not reuse alignment's root-weighted means:

```python
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig

dataset = MultimodalMixtureConfig.from_file("visual-sources.yaml")
means = dataset.estimate_mean_loss_weights(samples_per_source=128, seed=6198)
```

The input config must use the selected model's tokenizer/revision and source serialization.
Unit targets suffice when measuring per-source means. Calibration is not a full corpus replay,
but still incurs ordinary dataset preparation. Supply changed means explicitly through
`dataset.mean_loss_weight`. Unmasked 8K text has 8,191 next-token labels; masked mixed text
needs its own measured mean.

## Validation

Training saves source and optimizer diagnostics independently of checkpoint writes. Check
realized example/crop exposure, denominator and active-label weights, finite metrics, skips
and router drops. These diagnostics do not measure benchmark quality or exact gradient
allocation. For short performance measurements, use
`--trainer.callbacks.checkpointer.enabled=false` to disable checkpoint writes while preserving
parent and resume loading. `--trainer.no_checkpoints=true` is rejected because it also disables
checkpoint loading.

Use `src/scripts/eval/Vision-Align.py` for standalone `fast-text`, `decoded` and `academic`
evaluation of mixed checkpoints. The
[checkpoint benchmark commands](vision_alignment.md#checkpoint-benchmarks) cover invocation,
task-level recovery and CPU `--dry-run` / `--check-complete` modes. Save results
outside checkpoint directories and preserve the same tokenizer, panels, prompts, image controls,
generation limits, parsers and scoring across comparisons. Training callbacks remain extensible
through the standard configuration API; these benchmark suites are not training callbacks.

The decoded and academic panels are bounded diagnostics, not full official benchmark splits.
Inspect counting, pointing, OCR, captions and text quality separately, and check panel overlap
against the actual training inventory. The grounded-count validation split is excluded from
mixed training, but its images were included in alignment's scalar-count population; it is
not an end-to-end unseen-image holdout.

Run the focused tests:

```bash
pytest -q src/test/internal/vision_midtraining_test.py \
  src/test/internal/vision_midtraining_data_test.py \
  src/test/data/multimodal/text_only_collator_test.py
```

Short-run validation does not establish stability or quality across the full 50B budget.
Changed execution settings or model architectures need a distributed optimizer/checkpoint/resume
canary; configuration and CPU tests alone are insufficient. Source changes need realized-exposure
checks and quality evaluations, including longer continuations.
