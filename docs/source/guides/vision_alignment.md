# Vision alignment

Run `src/scripts/train/Vision-Align.py` with the standard internal experiment CLI.
The pipeline controller is `olmo_core.internal.vision_alignment_pipeline`; phase recipes and
visual sources are defined in `olmo_core.internal.vision_alignment` and
`olmo_core.internal.vision_alignment_data`. Configuration uses ordinary dataclasses and
dotted overrides. Data paths and launch defaults require access to Ai2 infrastructure.

By default, one Beaker job runs bridge, perception and joint training in order within the
same two-node allocation. Each phase uses a fresh distributed process and its own config,
optimizer and output folder. Select `--recipe.phase=bridge`, `perception` or `joint` to run
one phase independently. The final joint checkpoint can initialize mixed text/vision
midtraining.

## Training defaults

| Phase | Trainable components | Steps | Microbatch sequences per GPU |
| --- | --- | ---: | ---: |
| Bridge | Connector and six image-token input rows | 500 | 4 |
| Perception | Bridge components and vision encoder | 2,000 | 4 |
| Joint | Perception components and LM body; output projection frozen | 1,500 | 2 |

All phases use 8,192-token sequences, 128 sequences per global update, eight local detail
crops plus one overview per image, a 64-crop limit per packed sequence, a 48-example packing
buffer and eight loader workers. Packed examples have isolated attention and independent
positions. The per-image crop limit and packed-sequence crop limit are separate settings.

| Phase | Learning rates: connector / vision / LM | Warmup steps | Cosine horizon |
| --- | --- | --- | ---: |
| Bridge | `2e-4` / frozen / frozen | 100 | 250 |
| Perception | `5e-5` / `3e-6` / frozen | 100 / 250 | 2,000 |
| Joint | `2e-5` / `2e-6` / `1e-6` | 50 / 100 / 100 | 1,500 |

All schedules have a 10% learning-rate floor. Bridge remains at its `2e-5` floor from step
250 through step 500. Training duration and scheduler horizon are independent: changing
`trainer.max_duration` does not change the schedules. Set all component horizons and
warmups explicitly when changing a phase's schedule.

Joint allocates 16 text and 112 visual sequences per update. Its CE objective is a
0.35/0.65 weighted sum of the global text/vision mean losses, not a per-token or
gradient-norm split.

Bridge and perception disable router load-balancing (LB) loss while the LM is frozen. Joint
restores the original pretrained per-layer LB coefficients. Router z-loss and architecture are
preserved; no router-input RMS repair is applied. Execution settings of the LM (attention
backend, expert dispatch, recomputation) are not changed by the recipe: they come from the
text configuration described next.

The LM architecture and tokenizer, including its revision, come from the pretrained
checkpoint. Bootstrap supports OLMoDDP LMs, Molmo2-compatible tokenizers, reserved image-token
rows and untied input/output embeddings. Dense bootstrap and composable text replay are
not supported.

### Inheriting the text team's settings

Set `--recipe.text_config=/path/to/config.json` to the resolved configuration of the text
mid-training run the checkpoint belongs to (the `config.json` a trainer saves, or a `dry_run`
dump of the text workload). The LM configuration and every text-side training setting are then
inherited from it: the optimizer (`eps`, `sigma_factor`, `compile`, weight decay, betas,
gradient clipping), the train module (z-loss, `compile_model`, data and expert parallelism,
optimizer-state reset on load), the trainer's checkpointer and bookkeeping settings and its
monitoring callbacks, the loader worker count, and the launch image, install step, resources
and environment. The alignment phase changes only what
`olmo_core.internal.vision_alignment.MULTIMODAL_OVERRIDES` lists (phase learning rates and
schedules, per-group clipping, the multimodal batch and microbatch, checkpoint cadence, the
vision components and their bootstrap, the multimodal data and evaluator); a unit test keeps
that table exact. The LM in the text configuration must match the pretraining checkpoint,
whose weights are loaded.

Without `text_config`, the LM configuration is read from the checkpoint's own `config.json`
(legacy keys normalized, EMO routing cleared as text mid-training does) and the recipe's
legacy defaults apply (EP8, `eps=1e-6`, `sigma_factor=12`, z-loss `1e-4`).

### Document mode for recurrent language models

OLMo 3.5 language models mix Kimi Delta Attention (KDA) layers with a few attention layers.
KDA cannot apply an attention mask; it resets its state at document boundaries instead. When
the LM has such layers, the multimodal model isolates packed examples through document
boundaries derived from the loader's `example_ids` (one document per packed example, padding
in its own document), sends no attention masks or positions, and treats image tokens as
causal like text. Sibling-branch packing (`subsegment_ids`) needs a mask and is rejected, so
the recipe sets `annotation_sampling="one"` on the multi-annotation sources (PixMo points and
CoSyn pointing, which otherwise pack each of an image's annotations as a sibling branch): each
example keeps one annotation, sampled deterministically per seed, row and epoch, and those
sources use the matching calibration (`ALIGNMENT_ONE_ANNOTATION_MEAN_LOSS_WEIGHTS`). The
recipe also turns the experimental `kernel-fun` KDA kernels off for such models: they do not
take packed documents (`cu_seqlens`) and their support check crashes on them, so the FLA
kernels are used until that is fixed upstream.

## Configure and launch

Inspect the initial configuration and phase plan without loading weights or replaying data:

```bash
python src/scripts/train/Vision-Align.py dry_run align local \
  --recipe.pretraining_checkpoint=/path/to/pretraining/stepN
```

All-stage `dry_run` validates bridge and prints the later handoffs. Perception and joint
configs are built when their parent checkpoints exist, so this is not validation of the
future phase configs. For a single-phase configuration, include `--recipe.phase=PHASE`.

Submit all three phases:

```bash
python src/scripts/train/Vision-Align.py launch align ai2/holmes \
  --recipe.pretraining_checkpoint=/path/to/pretraining/stepN
```

With `--recipe.text_config`, the image, install step, node and GPU counts, shared memory,
priority, Weka mounts and environment come from the text run's launch configuration. The
legacy defaults are two eight-GPU nodes, EP8, urgent priority and 32 GiB shared memory. Both
use workspace `ai2/oe-olmo3p5-mt` with its default budget (`ai2/oe-other`), minimum runtime
8h, no log following, and the `jasonr_BEAKER_TOKEN` / `jasonr_WANDB_API_KEY` secrets. Change
these with `--launch.*` overrides. Common overrides apply to every phase; use an explicit
phase for phase-specific tuning. Inspect `launch.env_secrets` and configure the Beaker and
W&B secrets for the submitting account and target workspace. AWS credentials are not mounted
by default. Checkpoint-derived S3 text replay requires suitable credentials or an explicit
accessible storage mirror.

The launcher clones a remote git commit. `allow_dirty` does not upload local edits or
untracked files; unpublished changes require an explicitly frozen source deployment.

In an existing allocation, run all-stage `train` as a Python controller, not under
`torchrun`; it starts a fresh `torchrun` process for each phase. An explicit-phase `train`
uses the standard `torchrun` invocation. The worker count must match configured parallelism.
`prep`, `launch_prep` and `eval_checkpoints` require an explicit phase. `prep` constructs
and prepares the data loader; it is not a mandatory pre-launch audit.

## Checkpoint handoffs and resume

For run name `align`, the pipeline writes `align-bridge`, `align-perception` and `align-joint`
under `recipe.output_root`. Default handoffs are bridge step 500 to perception, then
perception step 2,000 to joint; joint ends at step 1,500. Saved configs always identify their
individual phase. All-stage mode rejects `recipe.parent_checkpoint` and a shared
`trainer.save_folder`; use `recipe.output_root` to choose the common output directory.

A phase is marked complete only after training and held-out validation at its final step
succeed. These are the small in-training checks, not external benchmark suites. Restarting
the same pipeline skips completed phases and resumes the active phase with full state.
A checkpoint alone does not mark a phase complete.

To run a phase independently, provide its parent checkpoint. The explicit-phase path uses
the exact supplied run name without adding a phase suffix:

```bash
python src/scripts/train/Vision-Align.py dry_run align-perception local \
  --recipe.phase=perception \
  --recipe.parent_checkpoint=/path/to/align-bridge/step500

python src/scripts/train/Vision-Align.py dry_run align-joint local \
  --recipe.phase=joint \
  --recipe.parent_checkpoint=/path/to/align-perception/step2000
```

A phase transition loads model weights and resets optimizer, loader and trainer state.
Resuming in the same phase's existing output folder restores full state, including packing
and RNG state. A restored step-zero checkpoint also skips fresh LM/vision initialization.
Parents must have compatible phase, model, tokenizer and pretraining ancestry.

`recipe.restore_pretraining_router_lb` restores the original per-layer coefficients in
joint by default. Set it to `false` to retain the parent's policy, or to `true` for explicit
restoration. `recipe.router_lb_loss_weight` instead sets a uniform coefficient and suppresses
automatic restoration. Explicit restoration and coefficient overrides are mutually exclusive.
Neither changes router z-loss or dispatch capacity.

## Visual sources and calibration

`dataset.sources` maps source names to dataset configs implementing `build(tokenizer)`.
`MultimodalSourceConfig` can apply prepared row selections. Bridge uses PixMo captions and
transcripts with loss-weight targets of 0.70/0.30. Perception and joint add PixMo pointing,
scalar counting, CoSyn pointing, OCR/document data and filtered VisualWebInstruct/Geo170K
alignment data. The visual recipe does not include Tulu.

Pinned FineVision sources verify their materialization manifest and loaded Arrow files.
The launcher caches successful byte verification under `recipe.work_dir/data-verification`;
unchanged files need only metadata checks on subsequent launches. Modified or replaced files
are reverified. Local runs can set `OLMO_CORE_DATA_VERIFICATION_CACHE_DIR` to a trusted shared
cache directory for the same behavior. This does not decode images or replay training data.

Targets describe supervised-loss weight, not image counts or input tokens. Source sampling
is proportional to `target_loss_mass / mean_loss_weight`. The supplied means use each
phase's tokenizer, prepared population, serialization and response weighting. When these
change, estimate new means with bounded sampling:

```python
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig

dataset = MultimodalMixtureConfig.from_file("my-sources.yaml")
means = dataset.estimate_mean_loss_weights(samples_per_source=128, seed=0)
```

Supply the source config, `dataset.target_loss_mass.SOURCE` and
`dataset.mean_loss_weight.SOURCE` together. A copy of the default artifacts in another
`recipe.artifact_root` keeps the supplied means when its build manifests are byte-identical
(`ALIGNMENT_ARTIFACT_MANIFESTS`); any other root needs explicit means. Calibration is not a
full corpus replay, but still incurs ordinary dataset preparation. Train/validation image
disjointness belongs in data preparation; retain the prepared selections when using the
default sources.

`recipe.sequence_length` updates training lengths without changing RoPE. Source serialization,
crop, packing or split changes require compatible calibration and a fresh data stream.
Loader fingerprints validate resume compatibility.

## Native text replay

Joint uses `PretrainingReplayConfig` to resolve the original LM's fixed-length numpy dataset.
It preserves saved file order, storage dtype, tokenizer, weighted mixtures, repetition
filtering and label-mask sidecars. Tokens are not retokenized, labels shift once, and text
examples carry no images. Storage URLs are not rewritten automatically.

The default holdout reserves 1,024 windows with seed 6198 and excludes them from joint
replay. It is not a holdout from LM pretraining. Automatic splitting requires unique,
unweighted fixed-length paths. Weighted or duplicate paths need explicitly disjoint
validation data and `recipe.text_validation_size=0`.

Set `dataset.sources.native_text_replay.dataset` to use an explicit dataset or storage
mirror, preserving file order, tokenizer, dtype, masks and filtering. Supply the matching
`dataset.mean_loss_weight.native_text_replay`: 8,191 for unmasked 8K replay; masked replay
requires calibration. This is a joint-only override; use explicit-phase mode rather than
applying it to all three phases. Check storage access separately from metadata-only validation.

Sequence allocation and objective weights are configured independently:

```python
config.data_loader.group_sequence_quotas = {"text": 16, "vision": 112}
config.train_module.loss_group_weights = {"text": 0.35, "vision": 0.65}
```

Source groups are derived from the configured sources unless explicitly overridden:
pretraining replay is text and visual adapters are vision. Groups pack independently and
retain calibrated source sampling within each group. Visual source targets set relative
sampling, while `loss_group_weights` sets the global objective coefficients.

Quotas count padded sequences, must sum to the global sequence count, and must be positive
multiples of DP size. With DP size 16, each rank receives one text and seven visual sequences.
Every group needs positive supervised mass on each update. Changes to groups, quotas or
DP size invalidate saved grouped-loader state.

## Validation

The training callback evaluates deterministic held-out CE on 64 examples per source at
startup, every 500 steps and at completion unless that step was already evaluated successfully.
Bridge/perception sources retain a 2,560-token
serialization cap and execute at 8K; joint uses 8K sources and rank evaluation batch size one.
Caption/transcript also use correct/wrong-image pairs matched for crop geometry and pooling
indices. Joint adds the replay holdout. These checks measure CE and image dependence, not
decoded accuracy; decoded and external benchmarks remain separate evaluations.

### Checkpoint benchmarks

`src/scripts/eval/Vision-Align.py` provides three independent runners. Each uses one
eight-GPU allocation and saves results separately from checkpoints:

| Suite | Definition |
| --- | --- |
| `fast-text` | Complete 26-task OLMES-fast panel, native completion interface, 2K context |
| `decoded` | Saved 448-example diagnostic panel covering counting, pointing, OCR and captions |
| `academic` | Saved selections for VQAv2, TextVQA, DocVQA, ChartQA, AI2D and A-OKVQA; 512 examples per task, with correct/shuffled/blank image controls |

Use a saved panel and matching tokenizer; the runners do not regenerate selections or
replay training-data inventories. The academic panel is a fixed validation subset, not
the full official benchmark splits. Its overlap annotations describe the manifest's
original training inventory, not necessarily the checkpoint being evaluated.

```bash
torchrun --standalone --nproc-per-node=8 src/scripts/eval/Vision-Align.py fast-text \
  --checkpoint /path/to/stepN --tokenizer /path/to/dolma2/tokenizer.json \
  --output /path/to/eval/fast_text.json

torchrun --standalone --nproc-per-node=8 src/scripts/eval/Vision-Align.py decoded \
  --checkpoint /path/to/stepN --panel-checkpoint /path/to/panel/checkpoint \
  --panel-file /path/to/panel.json --output-dir /path/to/eval/decoded

torchrun --standalone --nproc-per-node=8 src/scripts/eval/Vision-Align.py academic \
  --checkpoint /path/to/stepN --manifest /path/to/academic-manifest.json \
  --hf-cache /path/to/hf-cache/hub --output /path/to/eval/academic.json
```

All runners support CPU `--dry-run` and `--check-complete` modes. Run these with `python`
instead of `torchrun`. Install the `eval` extra for the pinned `ai2-olmo-eval==0.9.0`
fast-text harness and SciPy point matching. Saved panels and their referenced datasets
must be available independently of the source checkout.

Completed work is reused after preemption: whole tasks for fast text and academic, and
completed source/rank groups for decoded. An interrupted unit is repeated. To distribute
academic tasks over independent allocations, pass disjoint `--tasks` lists with the same
output path. Once all tasks finish, use `academic --merge` with the original arguments to
validate and assemble the full sheet on CPU. Replicas must not depend on a training leader.

Result formats distinguish the benchmark definition from its execution. Completion records
are reused only when both match; other result formats are not adopted as cache entries.

All phases save per-step metrics and preserve restored-step metrics before startup evaluation.
Permanent checkpoints are saved every 500 steps and temporary checkpoints every 50. Joint
retains three permanent checkpoints; bridge/perception retain two. Data errors fail immediately.

Run the focused recipe and component tests:

```bash
pytest -q src/test/internal/vision_alignment_test.py \
  src/test/internal/vision_alignment_bridge_test.py \
  src/test/internal/vision_alignment_perception_test.py \
  src/test/internal/vision_alignment_joint_test.py \
  src/test/internal/vision_alignment_data_test.py \
  src/test/data/multimodal/alignment_test.py \
  src/test/data/multimodal/pretraining_replay_test.py \
  src/test/data/multimodal/grouped_mixture_test.py \
  src/test/train/callbacks/multimodal_test.py
```

A new architecture or execution configuration needs a short distributed canary covering
initialization, frozen parameters, label masks, dispatch health and full-state resume.
`train_module.trim_microbatch_image_padding` is disabled by default; enabling it requires
valid collator counts, zero vision dropout and distributed validation.

Serialized configs identify classes by module path. Checkpoints with custom config or
callback classes require those modules when resuming training; preserve the corresponding
source deployment. Evaluation loads only the model and evaluation-relevant configuration.
