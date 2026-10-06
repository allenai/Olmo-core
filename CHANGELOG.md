# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v3.0.0](https://github.com/allenai/Olmo-core/releases/tag/v3.0.0) - 2026-09-30

### Added

- Added the native three-stage vision alignment recipe (bridge, perception, joint) for OLMoDDP language models as multimodal-only code: `MultimodalOLMoDDPModel` (an adapter that trains through the OLMoDDP train module's `input_embeddings` / `router_loss_div_factor` hooks), `MultimodalOLMoDDPTrainModule` (weighted response-only loss divisor, frozen-parameter checkpoints, native text-checkpoint loading), `MultimodalOLMoDDPOptimizer` (per-scheduler-group clipping, component gradient norms, partial master synchronization), the alignment data adapters, evaluators and the `Vision-Align` scripts. Text-only training code is unchanged. Two parts of the original #868/#869 runtime are not carried by this layer: the router token mask (router load-balancing and z-loss statistics include packed padding tokens; the bridge phase uses a zero load-balancing weight) and checkpointing of frozen expert weights under expert parallelism (saving raises `NotImplementedError`; the recipe runs with `ep_config=None` or trainable experts).
- Added `use_array_if_local` to `pack_documents_into_instances`, `segment_documents_into_instances`, `NumpyPackedFSLDataset` and `NumpyPackedFSLDatasetConfig`, forwarded to `iter_document_indices`. Set it to `False` to take document boundaries from the source metadata file instead of inferring them by scanning the token array for the EOS token. Inferring is only correct when every document is EOS-terminated: a producer that truncates documents and drops the terminator with the tail causes the affected document to merge with the one after it, and `LongDocStrategy.truncate` then keeps only the head of the merged span, so the following document never reaches training. Measured on an SFT cache, 97.98% of tokens reached instances via the inferred path versus 100.00% via the metadata file, with an identical maximum document length. Whenever the metadata boundaries are the effective source -- set explicitly, or because the source is a URL, for which `iter_document_indices` always reads the metadata -- `doc_lens` is derived from them rather than by rescanning the packed tokens for EOS, so the block-diagonal attention mask cannot merge an unterminated document into the one after it. The hazard is now documented on `iter_document_indices`. Default behavior is unchanged.
- Added opt-in paired SwiGLU backward and BF16-rounded weight-gradient accumulation for OLMoDDP experts, including Torch 2.13 support, checkpoint/recomputation coverage, and explicit backend and bucket-ownership guards.
- Added independent per-head Q/K norm gains and scalable softmax, EMO document-pool routing/global load balancing, and opt-in FP32 gradient-accumulation/reduce-scatter fast paths with explicit hardware/version guards.
- Extended hybrid MoE HF export for KDA, optional EMO and latent experts, per-head normalization gains, and scalable softmax, with exact tensor round-trip validation and legacy configuration migration.
- Added an optional, dependency-free checkpoint-ready notification callback for independent upload services. It does not upload or delete checkpoints.

### Removed

- Removed the `src/scripts/train/private-olmo.py` training script.

- Removed the `olmo_core.model_ladder` API, its internal CLI wrapper, all eleven ladder training scripts (including the standalone Gemma-like ladder), two Slurm launchers, and the API documentation. Existing ladder orchestration configs and commands require an earlier revision; this does not change the model or optimizer checkpoint formats. The internal experiment framework remains available.

### Fixed

- Metadata-backed packed datasets now include sidecar content hashes in packing-cache keys and dataset fingerprints, invalidating stale boundaries even after same-size corrections. Document lengths preserve EOS/BOS padding segmentation. Local array-backed defaults are unchanged (https://github.com/allenai/Olmo-core/pull/843).
- Apply opt-in Q/K gain expansion to eval-only and model-only DDP checkpoint loads, and reject forced expert assignments and biased KDA convolutions during HF export.
- Validate normalization throughout MoE HF exports, preserve attention-only gates and resolved EOS/padding IDs, and reject unsupported shared-expert routing before conversion.
- Reject MoE HF exports with incompatible Q/K normalization or inconsistent KDA output-norm epsilons instead of silently changing normalization behavior.
- Preserve the resolved tokenizer BOS ID in exported model and generation configs, including when the training config leaves BOS unspecified.
- Reject hybrid KDA HF exports with sliding-window attention and EMO exports with restricted evaluation pools in any routed layer. Disable scalable-softmax HF generation caching and reject explicit cache use.
- Require the CUDA 13 CuTe compiler for experimental KDA, report incompatible installs before training, and exercise the kernels in a dedicated Blackwell CI job.
- Keep legacy fused attention configs compatible when the new attention options are disabled. Reject unsupported scalable-softmax context parallelism and KV caching at setup.
- Assign EOS tokens to their preceding document for EMO routing, matching attention document boundaries.
- Preserve serialized tokenizer behavior during HF export, including source BOS settings when the training config leaves BOS unspecified.
- Preserve FP32 router probabilities and accumulation when combining BF16 experts in the HF reference model.
- Avoid graph breaks from no-op profiling decorators, including compiled router load-balancing collectives.
- `dispatch_flash_attn_4` passed `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q` and `max_seqlen_k` positionally. flash-attn 4 inserted a `qv` parameter at position 3 (present from ~`4.0.0b19` onward), which shifts every following argument by one, so `max_seqlen_q` — an `int` — lands where `cu_seqlens_k` is expected and the call fails with `AttributeError: 'int' object has no attribute 'shape'`. These are now passed by keyword; the parameter names are unchanged across flash-attn 4 releases, so this is correct against both old and new versions. Only reachable on Blackwell, where `has_flash_attn_4()` returns True.

### Changed

- `OLMoDDPModel.apply_ddp()` now raises `NotImplementedError` directing callers to `apply_dp()`, including under `python -O`. Removed its unreachable legacy DDP implementation.

- `OLMoDDPTrainModuleConfig.max_grad_norm` now overrides the optimizer clipping threshold when set, matching the train-module configuration API used by `TransformerTrainModuleConfig`. When unset, the optimizer threshold is retained. Previously this field was ignored, so old configs with differing values now use the train-module value. The supplied optimizer config is not mutated. Removed the unused `OLMoDDPTrainModule.max_grad_norm` constructor argument and attribute; clipping remains inside the DDP optimizer.

- Removed the unused, commented-out Beaker execution-unit helper and its commented call from the internal experiment setup.

- Migrated repository attention callers to explicit backend selection. Transformer factories now resolve legacy `use_flash` arguments into `backend` configs, and the nGPT factory accepts `attn_backend`. Deprecated inputs remain supported for old configs and callers, including explicit-backend precedence and sliding-window automatic selection.

- Replaced `FusedAttention` with `FusedAttentionV2` and migrated its applicable tests to V2. Legacy `AttentionConfig(name="fused")` configs still load through V2 with the same packed QKV parameter names and shapes, preserving model and optimizer checkpoint compatibility. Legacy fused RoPE is mapped to regular RoPE: floating-point results (especially low-precision RoPE gradients) differ, so resumed training is not numerically identical. `TransformerConfig.llama_like(fused_ops=True)` now selects V2 with regular RoPE when it previously selected fused attention.

- Raised the minimum supported PyTorch version to 2.10.0. Updated the stable Beaker images to PyTorch 2.10 with CUDA 12.8 and PyTorch 2.11 with CUDA 13.0, and expanded CI coverage to include PyTorch 2.10 and 2.11 with CUDA 12.8 and Docker builds through PyTorch 2.12 with CUDA 13.0.

## [v2.6.0](https://github.com/allenai/Olmo-core/releases/tag/v2.6.0) - 2026-08-11

### Added

- Added `KimiDeltaAttentionConfig.use_experimental_kernels` (**experimental**), which routes both the KDA chunk kernel and the layer's short convolutions to the CuTe/Triton kernels in the `kernel-fun` package, installed with the new `kernel-fun` extra. Both are drop-ins that fall back to FLA off their supported hardware (Blackwell for KDA, Hopper and up for the conv) and shapes.
- Added `KimiDeltaAttentionConfig` / `KimiDeltaAttention`, a Kimi Delta Attention sequence mixer built on Moonshot's released Triton kernel (via the `fla` extra). KDA uses a vector-valued decay per key channel plus a scalar delta gate per value head, and plugs into the existing sequence-mixer interfaces (packed-document short convolutions, initialization, context parallelism).
- Added the OLMoDDP fused Mixture-of-Experts training stack: the fused MoE transformer model and block (`OLMoDDPModel` / `OLMoDDPTransformerBlock` in `olmo_core.nn.ddp`, selected via `TransformerType.moe_fused_v2`), the `OLMoDDPTrainModule` train module with its fused distributed MoE optimizer (`OLMoDDPOptimizerConfig`) and `MultiGroupDistributedDataParallel` wrapper, and DP/EP/TP/CP/PP parallelism — including expert-parallel all-to-all over symmetric-memory / NCCL-RMA transports and a custom pipeline-parallel schedule (`CustomSchedule1F1BV` / `CustomScheduleInterleaved1F1B`) that reuses point-to-point buffers across micro-batches.
- Added the supporting kernels for the OLMoDDP stack under `olmo_core.kernels` (beta): grouped matmul (`grouped_mm` / `scaled_grouped_mm`), MXFP8 rowwise quantization (`OlmoMXFP8Tensor`), MoE token permute/unpermute, symmetric-memory / NCCL-RMA point-to-point communication, and a JIT loader for CUDA/C++ extensions; plus `FP8WeightStore` for fp32-main / FP8-model training and an experimental CPU activation-offload context.
- Added HuggingFace interop for the OLMoDDP model (beta): the standalone `olmo3moe` HF model (`Olmo3MoeConfig` + `modeling_olmo3moe`, since transformers has no OLMoE-MoE architecture) and olmo-core ↔ HF checkpoint conversion in `olmo_core.nn.hf` (`convert_olmo3moe_state_{from,to}_hf`), with per-layer-type RoPE and a `torch grouped_mm` routed-expert path.
- `TransformerTrainModule` now supports `eval_only=True` (build without an optimizer), so dense models work with eval-only workflows such as `Trainer.eval_checkpoints()` — previously only the pipeline and OLMoDDP train modules supported it.
- Added `Trainer.eval_checkpoints()` (and the `checkpoints_to_eval` config/trainer field): evaluate a list of checkpoint paths/globs with the configured `EvaluatorCallback`s without training. `TrainerConfig.build()` gained an `eval_only` pass-through for a uniform calling convention with the train-module config's `build()`.
- Added `max_checkpoints` parameter to `CheckpointerCallback` (default: 3) to limit the number of permanent checkpoints retained. Oldest checkpoints are removed automatically when the limit is exceeded. Set to `None` to keep all (previous behavior).
- Added `OutputDiscardCheckpoint`, an activation-recompute primitive for cases where the output of a checkpointed region dominates memory rather than its intermediates (e.g. precision casts, FFN up-projections). Forward runs under `no_grad`, the output's storage can be freed after downstream consumption, and a backward hook recomputes and rebinds the freed storage in place via a C++ `share_storage` extension (with a Python fallback for environments without a C++ toolchain).
- Added Qwen3.5 dense model configs (0.8B, 4B, 9B, 27B) with hybrid Gated DeltaNet + full-attention architecture.
- Added partial RoPE support via ``partial_rotary_factor`` on :class:`~olmo_core.nn.rope.RoPEConfig`.
- Added HuggingFace weight conversion for ``qwen3_5_text`` hybrid models.
- Added a configurable vision transformer encoder (`VisionTransformer`, configured via `VisionEncoderConfig`), vision-to-LM connector (`VisionConnector`), and `MultimodalLM` — a composite vision-language model that fuses image patch tokens into the LM token stream. Supports OpenAI CLIP, SigLIP, and SigLIP2 encoder variants with factory configs for all standard Molmo2 checkpoints.
- Added bidirectional image-token attention to `MultimodalLM` via a `token_type_ids` forward argument: image tokens attend to each other bidirectionally while text stays causal, matching HF Molmo2. Implemented as an `or_mask` threaded through `Transformer.forward` and the dense (`torch`) attention backend (OR'd onto the causal/sliding base); flash/TE backends raise rather than silently ignore it.
- Added `HFConverterCallback`, which can be used to convert models to huggingface format at the end of the training run.
- Trainer now records checkpoint save and load durations as `train/checkpoint_save_duration_s` and `train/checkpoint_load_duration_s` metrics.
- Added `PowerLR`, a power-law learning rate scheduler with linear warmup, power-decay phase (`lr = initial_lr * (current / warmup) ** b` for negative `b`, making the LR independent of the training horizon), and an optional linear decay tail. Registered as `"power_lr"`.
- Added `ComposableScheduler`, a piecewise LR scheduler built from `ComposableSchedulerStage` segments (linear/cosine interpolation between endpoint LRs) on an absolute time axis. Registered as `"composable"`. Note: `ComposableScheduler` ignores the `t_max` passed to `get_lr` and emits a once-per-instance `UserWarning` to that effect.
- Added `OverrideDecay`, a late-stage decay override usable on both `ComposableScheduler` and `SequentialScheduler` via an `override_decay` field. When `current >= override_decay.start`, the main schedule is interrupted mid-flight and the LR decays from the value the main schedule would have produced at `start` to a target LR over `duration` (linear or cosine). `SequentialScheduler` additionally warns that `t_max` is ignored once the override becomes active.
- Added `NvidiaProfilerCallback` (wraps a window of training steps in `cudaProfilerStart/Stop` + NVTX ranges for Nsight Systems) and `TorchMemoryHistoryCallback` (records CUDA memory history and dumps a snapshot pickle for https://pytorch.org/memory_viz).
- `SpeedMonitorCallback` now logs a `throughput/device/TFLOPs_per_GPU` metric and recognizes the RTX PRO 6000 device for peak-FLOPs / MFU estimation.
- `OLMO_RICH_LOGGING` can now explicitly enable *or* disable rich console logging (`0`/`false`/`no`/`off` disables it); previously setting it to any value only force-enabled rich logging.
- `init_distributed()` now bootstraps a minimal single-process environment (`RANK=0`, `WORLD_SIZE=1`, `MASTER_ADDR`/`MASTER_PORT`) when launch env vars are absent, so scripts can be run directly (without `torchrun`) for single-process debugging.
- Added a configurable `determinism_check` option to activation checkpointing (default `"default"`); set it to `"none"` to skip torch's recompute metadata check for opaque linear-attention kernels under `torch.compile`.
- Added an opt-in slow-batch-load warning to `SpeedMonitorCallback`: set `OLMO_BATCH_LOAD_WARN_SECONDS` to a positive number to log a warning whenever a step's batch-load time meets or exceeds it (an invalid value is ignored with a warning).
- Added `patch_torch_stream_custom_ops()` (called from `prepare_cli_environment()`), which registers a missing fake for PyTorch 2.11's `streams::sync_dealloc` op and installs a stable stream/event lookup fallback so `torch.compile` (notably compiled pipeline-parallel backward graphs) traces correctly. It no-ops on torch builds without `torch._dynamo.variables.streams`.


### Fixed

- `WandBCallback` and `CometCallback` now initialize before the checkpointer (via a higher callback `priority`) so that pre-train checkpoint saves no longer drop already-recorded metrics.
- `EvaluatorCallback` and `BatchSizeSchedulerCallback` now recognize `OLMoDDPTrainModule` (previously they only accepted `TransformerTrainModule`/`TransformerPipelineTrainModule`, so in-loop eval and batch-size scheduling aborted with an OLMoDDP run). `BatchSizeSchedulerCallback` also accepts the `OLMoDDPOptimizer` for its learning-rate adjustment.
- The CPU `Test` CI job now caches `HF_HOME` across runs so the HuggingFace roundtrip tests (Qwen3-0.6B, Gemma-3-270m) don't re-download their checkpoints every run.
- Excluded `mark_dynamic` from `torch.compile` tracing (`@torch.compiler.disable`).
- Clearer error messages (now include the offending values) when a rank batch size isn't divisible by the sequence length, or `max_target_sequence_length` isn't a multiple of `sequence_length`.
- S3 uploads/downloads now also retry on transient SSL errors (`ssl.SSLError`, botocore/urllib3 `SSLError`).
- Distributed checkpoint writes now clone each tensor before serialization to avoid accidentally writing the full backing storage of a view/shared tensor, with a guard that raises `OLMoCheckpointError` if a written tensor is unexpectedly larger than its `nbytes`.
- Fixed LM in-loop evaluator data-order drift across repeated runs by resetting loader bookkeeping before each pass and making deterministic reshuffling the default.
- Fixed Qwen3 implementation to match HuggingFace by applying RoPE in the input dtype (bf16) rather than upcasting to fp32.
- Fixed HF model conversion for Llama, Qwen3, and Gemma so that converted checkpoints roundtrip correctly.
- Fixed Beaker secret existence check to use the case-insensitive HTTP endpoint, avoiding spurious "secret not found" errors when secret names differ only in case.
- Fixed `Transformer.init_weights` so that under interleaved pipeline parallelism (e.g. `Interleaved1F1B`, `InterleavedZeroBubble`) the multiple model chunks owned by a single rank no longer initialize to identical parameters. Adds a `model_part_idx` kwarg incorporated into the seed as `model_part_idx * pp_size`.
- Disabled `torch.compile` tracing through `TEAttentionBackend.forward`, whose Python/pybind setup is not Dynamo-safe.
- Fixed `TransformerPipelineTrainModule.num_flops_per_token` returning `None` under pipeline parallelism. Each PP rank only holds its stage's layers, so summing FLOPs from `model_parts` undercounts the model. Capture `model.num_flops_per_token` as a bound method before `split_model` deepcopies and drops layers, then call it at metric time. On meta device (the standard PP init path) this has no memory cost.
- Fixed `GatedDeltaNet.num_flops_per_token()` to report training FLOPs (a factor of 3 for forward + dgrad + wgrad), matching the other sequence mixers; it previously reported forward-only work, undercounting GDN MFU/TFLOPs.
- Fixed attention FLOP accounting (`Attention` / `FusedAttention` `num_flops_per_token`) to count the exact causal (and sliding-window) attended `(query, key)` pairs. It previously ignored causal masking, overcounting full-attention layers by ~2x; reported model TFLOPs/MFU on the attention-compute term therefore drop relative to older logs.

### Changed

- Set transformers version to >= 5.4.0 for Qwen 3.5 and in sync with open-instruct
- Added a documented `deterministic` option to `LMEvaluator` and `LMEvaluatorCallbackConfig` so callers can opt out of fixed eval ordering when desired.

## [v2.5.0](https://github.com/allenai/Olmo-core/releases/tag/v2.5.0) - 2026-04-01

**This version was never published to PyPI.** The upload was rejected because the package
metadata contained a direct URL requirement (the `dion` extra), which PyPI does not accept.
The tag and GitHub release exist, but `pip install ai2-olmo-core==2.5.0` will not work — install
from the tag directly, or use v2.6.0 or later. Fixed in v2.6.0.

### Added

- Added gradient dumping support to `GAPMonitorCallback`. Set `dump_gradients=True` to save raw gradient tensors during training. Supports full distributed checkpoints and preview mode (`dump_gradients_save_first_n`).
- Added API reference and user guide for the `olmo_core.generate` module and interactive chat interface.
- Added support for in-loop perplexity evals with context parallelism (CP) and tensor parallelism (TP).
- Added documentation for verifying chat template settings before running evals after SFT.
- Added `olmo_core.data.composable` module.
- Added `PeriNormTransformerBlock`.
- Added exponential learning rate scheduler to `olmo_core.optim.scheduler`.
- Added internal Olmo3 32B midtraining and long context configs.
- MoE: Added `TrainModuleConfig` ABC
- Added a `MetricSaverCallback` which just saves metrics at specific intervals to JSON files in the `save_folder`.
- Added `fixed_steps` option to `Checkpointer` and `Evaluator` callbacks for configuring checkpoints/evals at specific step numbers.
- Added support for supervised finetuning.
- Added support for gated attention.
- Added support for no-global-rope ("GNoPE").
- Added support for MXFP8 Linear layers via torchao.
- Added support for tracking total flops.
- Added support for Gemma 3 models.
- Added support for Qwen3 models.
- Added support for Muon and Dion optimizers.
- Added support for Ulysses-style context parallelism.
- Added Beaker URL to Wandb logging.
- Added 60M, 14M, and 1M model sizes.
- `SpeedMonitorCallback` will log Chinchilla multiple number of tokens during training with a `TransformerTrainModule`.
- Added support for flash-attn 4 (CUTE implementation).
- Added `Callback.pre_log_metrics()` method.
- Added `SequenceMixer` base class that both attention and recurrent layers inherit from.
- Added `GatedDeltaNet` layer implementation.
- Added `InitMethod.fan_in` for per-layer fan-in initialization where each weight matrix uses `std = 1/√d_in`.
- Added `StabilityMonitorCallback` for detecting training instability via spike detection in loss and gradient norm.
- Added `gate` and `activation` parameters to `TransformerConfig.gemma3_like()`.
- Added another ladder script, with `--train-single` flag.
- Added `CuTeRMSNorm`, a CuTe-based RMSNorm implementation from the QuACK library.
- Added `lazy` option to `DownstreamEvaluatorCallbackConfig` for lazily loading each task which can decrease startup time.
- `TrainingProgress` (from `Trainer.training_progress`) now includes `current_tokens`, `bps`, `tps`, and `mfu` fields.
- `BeakerCallback` will include throughput metrics in the workload description.
- Added `olmo_core.io.deterministic_glob_directory` function.
- Added the option to cache the results of certain IO operations on remote files, like `get_file_size()` and `deterministic_glob_directory()` by setting the env var `OLMO_CORE_FS_CACHE_DIR` to a local directory.
- Added `eval_on_finish` option to `EvaluatorCallback`.
- Added the option to use a process pool instead of a thread pool when writing checkpoints.
- Added `max_document_length` and `long_doc_strategy` options to `NumpyDocumentSource` in composable data API.
- Mark ephemeral checkpoints with the `ephemeral` flag in their metadata.
- Added `ephemeral: Optional[bool]` flag the `Checkpointer.find_checkpoints()` for filtering.
- Added support for block-pattern based initialization of hybrid transformers. `TransformerConfig.block` now accepts a dict of named `TransformerBlockConfig`s, paired with a `block_pattern` list that controls per-layer block selection.
- Added optional `vocab_size` field to `DataCollator` for validating token IDs are in `[0, vocab_size)` before the batch reaches the model. Wired through automatically in both `NumpyDataLoaderConfig` and `ComposableDataLoaderConfig`.
- Added Olmo-hybrid official training configs and conversion script.
- Added new in-loop eval tasks: Generative QA BPB tasks, expanded MT-MBPP languages, and Science/Medical RC tasks.
- Added paged KV cache support to `FlashAttention4Backend` for inference on Blackwell (SM >= 10.0) GPUs.
- Added Code Fresh per-language perplexity evals

### Fixed

- Fixed `Transformer.get_rope_buffers()` crashing on non-rope attention mixers like `GatedDeltaNet`.
- Fixed A100 peak flops spec in `SpeedMonitorCallback` being 2x too low, which inflated MFU by 2x.
- Fixed `AttentionConfig.num_params()` overcounting QK norm parameters when using GQA/MQA with `use_head_qk_norm=False`.
- Fixed the peak learning rate in `src/scripts/train/OLMo3/OLMo3-32B-midtraining.py` to the correct one.
- Fixed type annotation issue in `NumpyInterleavedFSLDataset` where `_num_interleaving_exempt_instances` and `_num_interleavable_instances` were missing `Optional[int]` type hints, causing mypy type errors.
- Fixed bug in GPUMonitorCallback where it was using a Wandb reserved keyword, causing data to be unable to be visualized in the Wandb dashboard.
- Fixed the ConsoleLoggerCallback filtering to support the new prefix (gpu_memory) for GPUMonitorCallback.
- Avoid torch dynamo recompiles when intra-document masking enabled by marking `cu_doc_lens` and `max_doc_len` dynamic.
- Flops tracking for ParallelMLP and SWA layers.
- Fix overflow when too many global flops are computed.
- Ladder lmevaluator typo.
- Made some functions involved in data loading preprocessing more robust to race conditions.
- GAPMonitorCallback would raise an error if a local tensor shard had 0 elements.
- Fixed a bug where final metrics might not get logged.
- Fix failing test_build_world_mesh_cpu for pytorch 2.10.
- Fix failing convert_checkpoint_to_hf_test due by reducing total disk space required.
- Ensure all metrics have been logged and bookkeeping ops complete before writing a checkpoint.
- Fixed `self == InitMethod.*` comparisons in `Attention`, `FusedAttention`, and `GatedDeltaNet` init that should have been `init_method == InitMethod.*`, causing depth-scaled output projection init to never apply.
- Minor improvements to make checkpointing more robust.

### Changed

- Updated SFT documentation with alternative tokenization approach and tips for new base models.
- Renamed `olmo_core.distributed.utils.scatter_object()` to `broadcast_object()` for correctness.
- Updated stable torch version to 2.9.1, updated versions of underlying libraries in Beaker Images.
- `olmo_core.io.join_path()` now accepts an arbitrary number of components to join.
- All `olmo_core.nn` module configs now inherit from a common base class, `ModuleConfig`.
- Big changes to `olmo_core.model_ladder` API.
- Add ngram instance filter to olmo3_ladder.
- Upgraded to beaker-py v2.
- Now, we check `dist.is_initialized()` before calling `dist.init_process_group()` in `init_distributed()`.

## [v2.4.0](https://github.com/allenai/Olmo-core/releases/tag/v2.4.0) - 2025-11-20

### Added

- Added option to skip ranges of steps in the trainer.
- Send a Slack notification when a Beaker job appears to be stuck.
- Added `ignore_fingerprint_mismatch` parameter to `NumpyDataLoaderConfig` to allow resuming training from a checkpoint with a different dataset mix.
- Added helpful error messages when OLMo-mix-0625 files are not found, directing users to use OLMo-mix-0925 and the fingerprint override flag.
- Added `olmo_core.generate.chat` module to allow interacting with OlmoCore models without conversion to other formats.
- Added `GAPMonitorCallback` for monitoring gradients, activations, and parameters (GAP).
- Added official Olmo 3 7B and 32B pretraining scripts and data mix.
- Added official Olmo 3 7B and 32B midtraining scripts and data mix.
- Added official Olmo 3 7B and 32B long-context scripts and data mix.
- Added a `NoOpOptimizer` that does nothing, uses no memory, and can be used for debugging.
- Added official config for Olmo 3 32B.
- Olmo 3 model card and checkpoint manifests.

### Fixed

- Set missing `NCCL_NVLSTREE_MAX_CHUNKSIZE` env var that is now needed for running jobs on Augusta cluster.
- Fixed bug with `RemoteFileSystemReader` that caused excess memory usage.
- No longer overrides `random`'s RNG seed when building `SourceMixtureDatasetConfig`.
- Fix handling URLs in `olmo_core.nn.hf.checkpoint.save_hf_model` and in `examples/huggingface`.
- Fix potential NaN loss that can occur when using instance masking.
- Stability improvements developed while training Olmo3 32B.

### Changed

- Removed unused field in `YaRNRoPEScalingConfig`.

## [v2.3.0](https://github.com/allenai/Olmo-core/releases/tag/v2.3.0) - 2025-10-17

### Fixed

- Fixed parsing username+password git remote URLs in `launch.beaker` module.
- Fixed bug with default setup steps in `launch.beaker.BeakerLaunchConfig` when a branch can't be resolved.
- Cluster names in Beaker have changed.
- Fixed mixture rounding error with `SourceMixtureDataset`, which was previously causing samples to be repeated at the end of training.
- Don't DDOS Beaker from big jobs.
- A configuration error is now raised if you pass in a URL for the trainer or dataset's working directory.
  Previously the URL would just get mangled into a local path, leading to unexpected behavior.
- Fixed an issue where the `ConsoleLoggerCallback` would attempt to log before the first step.
- Only call `teardown_distributed_environment()` when training ends cleanly to avoid a hang for the duration of the distributed backend's timeout when there's an error from one rank.
- Fixed tensor parallelism issue with torch 2.8.
- More fixes for Beaker cluster names.
- `Callback.post_train()` will still be called even if the run is canceled before the dry-run batch.
- `GarbageCollectorCallback` will restore `gc` settings even when `Trainer.fit()` exits on an error.
- Make `move_to_device` blocking for MPS device to fix possible incorrect transfer of data from CPU to MPS.
- Fixed bug where `glob_directory()` would fail to match certain glob patterns.
- Added one more type of error to retry on when the Google Storage API throws it.
- Perform a garbage collection after checkpointing to avoid running out of CPU memory.
- Avoidable overflow error when using NumpyPackedFSLDataset.
- Fixed issue with NumpyFSLDatasetMixture + SourceMixtureDataset where not all instances would have the same sequence length.
- Attention backend will no longer default to flash in non-CUDA environments.

### Changed

- The `dir` option to `Trainer.maybe_load_checkpoint()` is now optional and defaults to the `save_folder`.
- Set `fused_linear_cross_entropy_loss accum_dtype` to fp32 in `LMHead`.
- Increased `NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS` from 10 minutes to 30 minutes.
- `SlackNotifierCallback` will now notify on checkpoint saved and post epoch events.
- `BeakerLaunchConfig.launch()` will now send Slack notifications by default when `follow=True` if the env var `SLACK_WEBHOOK_URL` is set.
- `src/examples/llama/` has been renamed to `src/examples/llm/`.
- Refactored eval task groups into `task_groups.py`
- The `use_flash` argument to the `Attention` classes is deprecated. Use `backend="flash_2"` instead.
- Refactored `NumpyDatasetConfig` by splitting it into a separate config per underlying dataset class.
- Refactored `internal/experiment` module to facilitate modifying datasets or supplying a fully custom `ExperimentConfig`.
- Simplified `SourceMixtureDatasetConfig` by removing redundant `sequence_length` and `dtype` fields.
- The `model_id` argument to `convert_state_from_hf` is deprecated. Conversion information is deduced from the model type.
- Refactored the example conversion scripts to/from HF, including decreasing false failures in validation.
- Small refactor to `source_mixture.py` to make it easier to define data mixes in yaml.
- Reorganized/cleaned up internal training scripts.

### Added

- Added CLI script `src/scripts/unshard.py` for converting distributed checkpoints to regular PyTorch or safetensors format.
- Added a custom block that does LayerNorm scaling.
- Added `OLMo-mix-0625-150Bsample` data mix.
- Added alias support to `DataMix` enum.
- Added the `HalfCos` learning rate scheduler.
- Added `CONTRIBUTING.md` guidelines.
- Added a lightweight, gantry-like Beaker launch CLI: `python -m olmo_core.launch.beaker`.
- Added [Beaker images with torch 2.8](https://beaker.allen.ai/orgs/ai2/workspaces/OLMo-core/images?searchQuery=tch280). There is `olmo-core-tch280cu128-2025-09-18` and `olmo-core-tch280cu129-2025-09-18` for CUDA 12.8 and 12.9, respectively.
- Added TransformerEngine to Docker images and a TransformerEngine attention backend.
- Added `Callback.close()` method, which is always called when exiting `Trainer.fit()`.
- Added flash-attention 3 to Docker images, added `flash_3` attention backend.
- Added support for sliding window attention to the Torch attention backend. Performance is not optimized, so other backends should be preferred.
- Added `RoPEScalingConfig.to_hf_config()` for each RoPE scaling method to support automatic conversion to HuggingFace format.
- Guide to dataset mixing in `docs/source/guides/data_mixing.rst`.
- Added support for converting FlexOlmo models (with both dropless and default MoEs) between OLMo Core and HF formats.
- Added `olmo3_7B` model config.
- Added additional internal configuration tools.
- Added a new named data mix that we used for the 32B run
- Added the ability for `GenerationModule` to load multiple checkpoints at once and average them.
- Added internal OLMo3 7B midtraining config.
- Added internal OLMo3 7B midtraining and long-context configs.
- Added ability to convert OLMo3 models to/from HF format with support for rope scaling configs.
- Added the `WSDS` (Warmup-Stable-Decay-Simplified) learning rate scheduler.
- Added a script that can pull out a single training batch from a training job

## [v2.2.0](https://github.com/allenai/Olmo-core/releases/tag/v2.2.0) - 2025-08-26

### Added

- Added option to set LR scheduler based on tokens instead of steps (e.g. `--train_module.scheduler.units=tokens`).
- Added a "packed" numpy FSL variant that packs documents into sequences using the best-fit-decreasing bin packing algorithm following the work from [Fewer Truncates Improve Language Modeling](https://arxiv.org/pdf/2404.10830).
- Added module `olmo_core.testing`.
- Added a "interleaved" numpy FSL variant that interleaves several documents into sequences following the work from [LongSkywork: A Training Recipe for Efficiently Extending Context Length in Large Language Models](https://arxiv.org/pdf/2406.00605).
- Added sliding window attention as a feature
- Added `BatchSizeSchedulerCallback` for setting a batch size schedule over the course of a training run.
- Added optional `TrainModule` method, `.pre_train()`, which runs right after `Callback.pre_train()`.
- The `BeakerCallback` will save the config and Python requirements to the results dataset.
- Added `from_file` method to `Config` class.
- Added in-loop evals for OLMES basic skills eval
- Added in-loop fast MCQA for in-loop evals and translated MBPP tasks
- Added in-loop few-shot HumanEval BPB
- Added `fast` and `full` in-loop recommendations, where `fast` is a roughly 2-3x faster subset of `full`
- Added support for converting to HF models in lower precisions.
- Added support for headwise QK norm.
- Add BOS token in in-loop evals, when specified by the tokenizer (`ai2-olmo-eval==0.8.4`)
- Add support for BOS token matching EOS token for intra-document masking in FSL numpy datasets.
- Added option to allow profiler to record on multiple ranks.
- Added support for accessing Google on non-Google clusters via auth with service account keys.
- Added an example script for launching an SFT job.
- Added support for revisions in `convert_checkpoint_from_hf.py` and the `load_hf_model` method of `olmo_core.nn.hf.checkpoint`.
- `foreach` support in `SkipStepAdamW`.
- Added `budget` mode for activation checkpointing configuration.
- Added `io.remove_file()` and `io.glob_directory` functions.
- Added ABF, PI, and YaRN rope scaling strategies.
- Added a script to compare two WandB runs
- Added `namespace` option to `nn.buffer_cache.BufferCache`.
- Added the option to configure `head_stride` for context parallelism with ring-flash-attn.
- Added the option to group multiple npy source files together for packing with the packed FSL dataset by setting `source_group_size` to an integer greater than 1.
- Added `load_optim_state: Optional[bool]` option to `Trainer.load_checkpoint()`.
- Added `GenerationModule` for Olmo-core native autoregressive generation with support for kv caching.
- Added optional hostname constraints for beaker experiments on Google clusters.

### Changed

- Output of `LMHead` when `labels` is passed as input is now a 4-tuple instead of a 3-tuple, with `(logits, loss, ce_loss, z_loss)`, where `loss` is the combined loss (`ce_loss + z_loss`).
- The `ConfigSaver` callback will automatically set the config to save for other callbacks (`WandBCallback`, `CometCallback`, and `BeakerCallback` as of now).
- Fixed bug causing slow evals in BPB/RC in-loop evals due to fast MC
- Changed default precision of converted HF models in `src/examples/huggingface/convert_checkpoint_to_hf.py` to bfloat16.
- Changed default cluster to `saturn` in `src/examples/llama/train_launch.py`.
- Made some beaker secrets optional for internal experiments.
- Changed `SlidingWindowAttentionConfig` to improve clarity.
- Changed the default Beaker budget

### Fixed

- Modify `TokenizerConfig.from_hf()` to fallback to tokenizer_config.json if config.json is not found.
- Fixed loading checkpoints with missing keys from transformer train modules using torch 2.7.
- Made MoE load balancing loss more robust.
- Fixed a bug with `ReorderedNormTransformerBlock` when using fine-grained FSDP wrapping and activation checkpointing together.
- Fixed an issue preventing tensor parallelism from working with `LMHead` when using the "fused_linear" loss implementation.
- Fixed a bug with `LMHead` when using "fused_linear" loss implementation where the `ce_loss` output included the `z_loss` added to it.
- Fixed training on single GPU when using a `SkipStepOptimizer`.
- Fixed the initialization of the `CosWithWarmupAndLinearDecay` learning rate scheduler
- Ensured eval tasks are sorted to maintain the same order across ranks (the cookbook was configuring these in an unsorted way).
- W&B callback uses working directory instead of save folder for local cache.
- Reset speed monitor callback after changing batch size.
- Fixed parallelism compatiblity between cp + tp and cp + pp and added test to catch regressions.
- Ensure sharded parameters are initialized differently on separate ranks.
- Fixed fingerprinting for FSL datasets
- Fixed bug where `step` state in `SkipStepAdamW` was not incremented, biasing the optimizer steps. Added option to restore the bug for backwards compatibility.
- Removed `sklearn` from upstream dependency `ai2-olmo-eval`.
- Made removing ephemeral checkpoints more robust.
- Made running bookkeeping operations more robust.
- Ensure RoPE modules with different settings use a unique sub-cache for their buffers.
- Fixed bug with context parallelism where every transformer block would use the same RoPE buffers even if their RoPE was configured differently.
- Fixed MFU computation to work with FSDP, corrected some device specs.
- Optimization: avoid redundant calls to `model.train()` in `TransformerTrainModule`.
- `NumpyDatasetConfig.expand_glob` now works with remote directories.
- Fixed Attention block sharding when TP and head-wise QK norm are both applied.
- Added RoPE scaling configs to `rope` module's exports.

## [v2.1.0](https://github.com/allenai/Olmo-core/releases/tag/v2.1.0) - 2025-04-14

### Added

- Added 50B Dolmino 11/24 mix.
- Added support for auxiliary-loss-free MoE load-balancing, similar to DeepSeek-v3. You can activate this by setting `bias_gamma` to a non-zero float in your `MoERouter` config.
- Added support for sequence-level MoE load balancing loss.
- Compatibility with B200s.
- Added support for `warmup_fraction` as an alternative to `warmup_steps` in all schedulers, allowing warmup to be specified as a fraction of total training steps.
- A better config for the 1B model, ported from the old OLMo trainer.
- Added `auto_resume` option to `CometCallback` for resume an existing run.
- (BETA) Added methods `load_hf_model` and `save_hf_model` for saving supported OLMo Core models to HF transformers format.
Also added lower-level methods for converting state between the formats.
- Added the ability to run the evaluator callback on `.pre_train()` by setting `eval_on_startup=True`, and to cancel the run after the first time evals run by setting `cancel_after_first_eval=True`.
- Added support for label mask files with numpy FSL datasets.
- Added a `git` configuration to `BeakerLaunchConfig`.

### Changed

- `TransformerTrainModuleConfig` can now be used to build a `TransformerPipelineTrainModule` by adding a `pp_config` spec. This makes the `TransformerPipelineTrainModuleConfig` redundant, but it will be kept around for backwards compatibility until the next major release.
- Several state dict methods in `TrainModule` now take an `optim` option, which can disable the use of optimizer state.
- Updated `Float8Config` for latest version of `torchao`.
- Undo a fix applied to `olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture` that was generating a mismatch between the shape of instances in the dataset and the shape of instances in the data loader.
- Made the 1B and 7B scripts more similar to each other.
- Changed underlying logic and top-level arguments of `convert_checkpoint_from_hf.py` and `convert_checkpoint_to_hf.py`.
- Beaker experiments launched with the `BeakerLaunchConfig` will now log with ANSI colors enabled.

### Fixed

- Fixed calculation of total steps based on epochs at the end of a training job.
- Fixed a bug where the trainer might try to save a duplicate final checkpoint if the run that already completed was restarted.
- When submitting a Beaker job from a branch that's tracking a GitHub fork, Olmo-core now instructs Beaker to pull from the fork instead of from the main repo.
- Made Beaker image resolution more robust.
- Having `t_max` overrides in the default model configs is confusing and error prone, so we removed them.
- Beaker launcher will only clone a single branch at runtime when possible, which can be much faster.

## [v2.0.1](https://github.com/allenai/Olmo-core/releases/tag/v2.0.1) - 2025-03-18

### Added

- Added information about the official 32B training run.
- Added information about the official 32B anneal training run.
- Added automatic support for LL128 when running on Augusta.
- Added information about 32B training logs.

### Fixed

- The official config for the 32B had unrealistic batch size settings.
- Ignore `group_overrides` for frozen parameters instead of throwing an error.
- Bump `ai2-olmo-eval==0.7.1`, which fixes makes the in-loop evaluation consistent with OLMES by removing [a bias](https://github.com/allenai/OLMo-in-loop-evals/pull/6)

### Removed

- Removed the "fused" cross-entropy loss variant. It had a bug and consistently under-performed the native PyTorch version when compiled. See [Post Incident Report: bug with fused CE loss](https://docs.google.com/document/d/1IK6q2gX6mH7eQO_IItCZAYYlm4g4htL4mNWbTQuPKf4/edit?usp=sharing) for more information.

## [v2.0.0](https://github.com/allenai/Olmo-core/releases/tag/v2.0.0) - 2025-03-12

This major release introduces a few breaking changes. We've provided more information here: [Olmo-core v2 design and upgrade guide](https://docs.google.com/document/d/1LvANhNzA-MdtiD2pLniLTqB9wxSSuqY435WuJIADeFM/edit?usp=sharing).

### Added

- Added `TrainModule` abstraction with `TransformerTrainModule` implementation, which encapsulates both a model and optimizer.
- Added `namespace` argument to `Trainer.record_metric()`.
- Added support for context parallelism.
- Added support for expert parallelism with MoE models.
- Added in-loop evals for Minerva, GSM, HumanEval, MBPP (`ai2-olmo-eval==0.7.0`)
- Added `CosWithWarmupAndLinearDecay` learning rate scheduler
- Added `WSD` learning rate scheduler
- Added `RunDuration` in `model_ladder` to configure training durations in terms of Chinchilla multipliers.

### Changed

- The `Trainer` now takes a `TrainModule` instead of a model and optimizer, and several configuration options have been moved to `TransformerTrainModule`, including `rank_microbatch_size`, `fused_loss`, `compile_loss`, `z_loss_multiplier`, and `autocast_precision`.
- Several `TransformerModelConfig` options have been to `TransformerTrainModule` / `TransformerTrainModuleConfig`, including `dp_config`, `tp_config`, `float8_config`, and `compile`.

### Removed

- Removed the following callbacks: `MoEHandlerCallback`, `SchedulerCallback`, `MatrixNormalizerCallback`, `GradClipperCallback`, and `Float8HandlerCallback`.
  The functionality from all of those callbacks has been moved to the `TransformerTrainModule` class.
- Removed the callback methods `.pre_eval_batch()` and `.post_eval_batch()`.

### Fixed

- Fixed the model ladder code when training on mps or cpu device

## [v1.9.0](https://github.com/allenai/Olmo-core/releases/tag/v1.9.0) - 2025-03-10

### Fixed

- Ensure certain optimizer param group fields are not overridden by the values in a checkpoint.

### Added

- Added `instance_filter_config` field to `NumpyDatasetConfig`.
- Added conversion script for OLMo 2 checkpoints to Huggingface format.
- Added `BeakerCallback`.
- Added logging for in-loop eval throughput

### Fixed

- Ensure certain optimizer param group fields are not overridden by the values in a checkpoint.
- Fixed issue where non-zero ranks would report partially-reduced values for training metrics.

## [v1.8.0](https://github.com/allenai/Olmo-core/releases/tag/v1.8.0) - 2025-01-29

### Added

- Added support for tensor parallelism. See the `TransformerConfig` class for usage.
- Added more downstream tasks from the model ladder.
- Added `io.copy_dir()` function.
- Added new LR schedulers: `LinearWithWarmup`, `InvSqrtWithWarmup`, `ConstantWithWarmup`, `SequentialScheduler`.
- Added option to pre-download checkpoint files from remote storage before trying to load a checkpoint.
- Added a callback for sending Slack notifications.
- Makes the MPS device work on Apple Silicon
- Added `SkipStepAdamW` optimizer.
- The trainer can load model-only checkpoints now.
- Added the option to throttle checkpoint uploads to one rank from each node at a time.
- Added support for logging rich Table objects as text in source mixture datasets.
- Added `unshard_strategy` parameter to `unshard_checkpoint()` function in `olmo_core.distributed.checkpoint`.
- Added function `load_keys()` to `olmo_core.distributed.checkpoint`.
- Added support for low precision optim state in `SkipStepAdamW`.

### Changed

- Changed storage of shared shard state in sharded checkpoints from smallest shard to lowest rank (normally 0).
- Changed how the trainer handles loading a checkpoint when `load_path` is provided. Now `load_path` is only used if no checkpoint is found in the `save_folder`.

### Fixed

- Added missing `weights_only=False` argument to fix loading train checkpoints with newer versions of PyTorch.
- Fixed bug where GCS upload does not retry on transient failures.
- Fixed bug where source mixture datasets were truncating source files instead of randomly sampling.
- Fixed bug in source mixture datsets where sampling from small npy files raised an mmap exception due to 0 instances in the sampled index.

## [v1.7.0](https://github.com/allenai/Olmo-core/releases/tag/v1.7.0) - 2024-11-27

### Added

- Added `key_mapping` argument to `olmo_core.distributed.checkpoint.load_model_and_optim_state()`
  for loading checkpoints with different key names.
- Added `load_key_mapping` field to the trainer, same idea as the new `key_mapping` argument above.
- Added an implementation of nGPT called `NormalizedTransformer`.
- Added an example showing how to convert a HuggingFace Llama 3.2 checkpoint into the right format for Olmo-core.
- Added an API for scaling RoPE embeddings.
- Added a `ModelLadder` API.

### Changed

- The `w_out` and `norm` top-level children of the `Transformer` model are now wrapped together in an `lm_head` module. Training scripts will have backwards compatibility with older checkpoints due to the `load_key_mapping` explained above.

### Fixed

- (Optimization) Mark model input sizes as dynamic for `torch.compile()` to avoid recompile during evals or variable-sequence / batch size training. This doesn't seem to hurt throughput.
- Made HTTPS and GCS IO functions more robust.
- Fixed a bug where we were always getting dolma2 tokenized validation data when generating config with DataMix.v3_small_ppl_validation.

## [v1.6.3](https://github.com/allenai/Olmo-core/releases/tag/v1.6.3) - 2024-11-15

### Added

- Added `olmo_core.distributed.checkpoint.get_checkpoint_metadata()` function.
- (BETA) Added flag to compile the optimizer step. So far only tested with AdamW. May not work with other optimizers.

### Fixed

- Old ephemeral checkpoints won't be removed until after the latest ephemeral checkpoint is saved successfully.
- Made GCS uploads more robust.
- Fixed single-node training on Google Augusta cluster.
- `numpy.random.dirichlet()` does not always sum to 1.0, so allow for a small tolerance in validating domain weights.

## [v1.6.2](https://github.com/allenai/Olmo-core/releases/tag/v1.6.2) - 2024-11-08

### Added

- Added option to disable `GarbageCollectorCallback`, not that you'd want to do this usually, but I needed to run an experiment to show how important that callback is.

### Fixed

- Fixed a bug where some default callbacks could be added twice if given a different name by the user.
- Fixed a bug where some `Trainer` bookkeeping tasks may not complete before `.fit()` returns.

## [v1.6.1](https://github.com/allenai/Olmo-core/releases/tag/v1.6.1) - 2024-11-06

### Added

- Added `retries` field to `BeakerLaunchConfig`.
- Allow running on Augusta cluster with existing train scripts.
- Added `olmo_core.utils.logging_configured()` function to check if logging has been configured.

### Fixed

- Fixed a potential distributed deadlock bug when training without a separate CPU-only bookkeeping backend.
- Removed some unnecessary host-device syncs in `olmo_core.distributed.utils`.
- Added `Trainer(Config).async_bookkeeping` field to toggle async bookkeeping.

## [v1.6.0](https://github.com/allenai/Olmo-core/releases/tag/v1.6.0) - 2024-11-01

### Added

- Added option to compile the trainer's loss function (`Trainer.compile_loss`).
- Added `SourceMixtureDataset` for composing a training mixture based on ratios of source datasets.
- Added `NumpyFSLDatasetMixture` for constructing a `NumpyDatasetBase` from a `SourceMixtureDataset`. Note this is only supported for FSL datasets.
- Added tests for `SourceMixture*` and `NumpyFSLDatasetMixture`.
- Added `DownstreamEvaluatorCallbackConfig` class for running in-loop downstream eval via [OLMo-in-loop-evals](https://github.com/allenai/OLMo-in-loop-evals).

### Changed

- Moved some types into `olmo_core.data.types` to avoid some circular dependencies.

### Fixed

- Made GCS client more robust by automatically retrying timeout errors for most operations.

## [v1.5.0](https://github.com/allenai/Olmo-core/releases/tag/v1.5.0) - 2024-10-23

### Added

- Added Google Cloud support for `list_directory()` and `clear_directory()`.
- Added `CometCallback` for logging training runs to Comet.ml.
- Added `DataMixBase` class, to allow extending to new data mix groups.
- Added support for MoE-based models.
- Added method `DataLoaderBase.get_mock_batch()`.
- Trainer now starts with a dry-run of a fake batch created by `DataLoaderBase.get_mock_batch()`.
- Added `Callback.pre_backward()`, `.pre_eval_batch()`, and `.post_eval_batch()` methods.
- Added `Trainer.model_forward()`, `.get_losses()`, and `.eval_batch()` methods.
- Added a new `TransformerActivationCheckpointingMode`, "selected_ops" (requires torch 2.5 or newer).

### Changed

- `BeakerLaunchConfig.setup_steps` should now include steps to clone your repo (which it will by default). This change allows support for private repos.

### Fixed

- `prepare_cli_environment()` now calls `add_cached_path_clients()`.
- Removed an unnecessary host-device sync.

## [v1.4.0](https://github.com/allenai/Olmo-core/releases/tag/v1.4.0) - 2024-10-02

### Changed

- Updated default layer norm epsilon for OLMo models from `1e-5` to `1e-6` to match latest model.
- Renamed `FSLDataLoader` to `NumpyFSLDataLoader`.
- Renamed `VSLDataLoader` to `NumpyVSLDataLoader`.
- The trainer now takes a `data_loader: DataLoaderBase` instead of a `dataset: NumpyDatasetBase`.

## [v1.3.2](https://github.com/allenai/Olmo-core/releases/tag/v1.3.2) - 2024-09-27

### Added

- Added `Config.validate()`, `Config.replace()`, and `Config.apply()` methods.
- Trainer now records sequence length as a metric.

### Fixed

- Ensure additional cached-path clients are added in the process pool workers from some dataset preparation methods.
- Fixed `label_mask` tensor created by `NumpyPaddedFSLDataset`.
- Removed redundant warning messages about CUDA alloc retries.
- Fixed non-deterministic deadlock bug with async checkpointing.

## [v1.3.1](https://github.com/allenai/Olmo-core/releases/tag/v1.3.1) - 2024-09-26

### Fixed

- Fixed the name given to evaluator metrics logged.

## [v1.3.0](https://github.com/allenai/Olmo-core/releases/tag/v1.3.0) - 2024-09-26

### Added

- Added `torchao` to the Docker/Beaker images.
- Added support for `torchao` `float8` training via the `Float8HandlerCallback`.
- Added `Callback.post_attach()` method.

## [v1.2.0](https://github.com/allenai/Olmo-core/releases/tag/v1.2.0) - 2024-09-25

### Added

- Added support for wildcards in `OptimGroupOverride.params`.
- Added `NumpyPaddedFSLDataset` variant.
- Added `Evaluator` class and `EvaluatorCallback` for in-loop evals.
- Added `v3-small-ppl-validation` data mix.

### Fixed

- Fixed bug with data loader when using threading.

## [v1.1.0](https://github.com/allenai/Olmo-core/releases/tag/v1.1.0) - 2024-09-18

### Added

- Added support for changing train sequence length when loading a checkpoint.
- Added support for sequence length warm-up during training via the callback `SequenceLengthSchedulerCallback`.
- Added support for variable sequence length (VSL) datasets and VSL curriculums as introduced in ["Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum"](https://arxiv.org/pdf/2405.13226).
- Added `Lion` and `SkipStepLion` optimizers.
- Added `init_seed` argument to `Transformer` and `TransformerConfig`.

### Changed

- Renamed `MemMapDataset` to `NumpyFSLDataset`.
- Batch size is now specified in tokens, not instances.

## [v1.0.6](https://github.com/allenai/Olmo-core/releases/tag/v1.0.6) - 2024-09-05

### Added

- Added "selected_modules" transformer activation checkpointing mode.
- Added `OLMo-1B.py` official training script.
- Added `OLMo-13B.py` official training script.
- Added `Trainer.get_metric()`, `.get_loss()`, and `.get_zloss()` methods.
- Added `io.copy_file()` function.
- Added `ProfilerCallback` for profiling/tracing the training loop with PyTorch `profiler` module.
- Added an "L2 norm" metric reduce type.

### Fixed

- Made reducing metrics more numerically stable with large world sizes.

## [v1.0.5](https://github.com/allenai/Olmo-core/releases/tag/v1.0.5) - 2024-09-03

### Fixed

- Fixed bug with checkpointer callback searching for existing ephemeral checkpoints when the checkpoint folder doesn't exist.
- Checkpointer callback won't collect existing ephemeral checkpoints that were saved after the checkpoint that was loaded from.

## [v1.0.4](https://github.com/allenai/Olmo-core/releases/tag/v1.0.4) - 2024-09-01

### Added

- Added `Trainer.save_checkpoint()` and `Trainer.save_checkpoint_async()` methods.
- Added `Callback.post_checkpoint_saved()` and `Callback.post_checkpoint_loaded()` methods.
- Added `ConfigSaverCallback`.
- Added `MemMapDataset.fingerprint` property.

### Changed

- The `work_dir` argument to `TrainerConfig` now defaults to `save_folder` is `save_folder` is a local path, otherwise a temporary directory with the same name as the basename of the `save_folder`.
- The `seed` argument to `prepare_training_environment()` is now optional.

### Fixed

- Fixed setting the right env vars for single node training on Jupiter.

## [v1.0.3](https://github.com/allenai/Olmo-core/releases/tag/v1.0.3) - 2024-08-30

### Added

- Add `Trainer.hard_stop` field.
- The trainer now catches `SIGTERM` and marks the run as canceled.
- Added `CheckpointerCallback.remove` strategy for configuring which old checkpoints found in the save folder are removed.
- Added `ReorderedNormTransformerBlock` implementation.
- Added `WandBCallback.notes` field.

### Fixed

- Fixed bug with how command arguments were expanded by `BeakerLaunchConfig`.

## [v1.0.2](https://github.com/allenai/Olmo-core/releases/tag/v1.0.2) - 2024-08-29

### Added

- Added support for unsharding model state into `safetensors` format with `olmo_core.distributed.checkpoint.unshard_checkpoint(..., use_safetensors=True)`.
- Added `data.TokenizerConfig` config class and `data.TokenizerName` enumeration.
- Added data mixes with `data.DataMix` API.
- Added `block_idx` attribute to the `TransformerBlock` class.
- Added `init_method` option to `Transformer` for controlling how the weights are initialized.

### Fixed

- Fixed `list_directory` for remote folders.

### Changed

- Callbacks now have to have a name assigned.

## [v1.0.1](https://github.com/allenai/Olmo-core/releases/tag/v1.0.1) - 2024-08-26

### Fixed

- Fixed a bug with resetting the initial LR in optimizers after a loading a checkpoint.

## [v1.0.0](https://github.com/allenai/Olmo-core/releases/tag/v1.0.0) - 2024-08-26

### Added

- Ported, refactored, and optimized the modeling and training from the OLMo repo while fixing several bugs. Introduces a new highly efficient yet customizable trainer and a standard API for launching jobs directly to Beaker from a Python script.

## [v0.1.0](https://github.com/allenai/Olmo-core/releases/tag/v0.1.0) - 2024-06-11

### Added

- Initial release.
