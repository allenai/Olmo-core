# LeakySFT implementation and frozen tests

The test files in `src/test/leaky_sft/` were written before the implementation.
`tests.sha256` records their pre-implementation hashes; they must remain unchanged
when evaluating this implementation. `evidence/red.txt` records the initial run:
**43 failures, 9 CUDA skips**. All failures were missing functionality, not collection errors.

## Objective and configuration

Set `train_module.cpt_loss_scaler=0.1` on a standard TransformerTrainModuleConfig.
For valid next-token targets, the objective is

    (sum(output CE) + alpha * sum(context CE)) / (N_output + alpha * N_context)

Explicit zero is answer-only SFT with context CE diagnostics. One is all-token
language modeling on the same corpus and attention layout. Intermediate values
are per-token relative weights, not the fraction of aggregate loss from context.
`None` (the default) preserves the old training path and its normalization.

Each instance must supply `loss_valid_mask`, independent of the answer `label_mask`.
The dense whole-document packer, landmark packer, and raw concat/chunk source now
supply it. Other instance sources must explicitly implement the same contract;
LeakySFT refuses to infer validity from an SFT mask or special token IDs. The
standard collator and composable loader preserve validity. Real occurrences of
special token IDs remain valid; inserted landmarks and padding do not. The
landmark packer's optional exclusion of landmark predictors is respected.

Labels and masks shift together. When `doc_lens` is present, predictions across
isolated documents are excluded. This correction also applies to the explicit
zero-scaler control; it can differ from legacy SFT if legacy masks supervised a
new document's first token. Keep document masking enabled for packed examples.
Without document metadata, the batch is treated as a continuous sequence.

The answer/context loss sums share one transformer forward. The linear-CE stage
runs on disjoint groups, allowing the existing fused kernel to accept a scalar
upstream gradient for each group. At zero, context CE is computed without
building its backward graph. Empty groups have finite differentiable zero loss.
The regular head can return logits; fused loss does not materialize vocabulary
logits for the full sequence.

Normalization uses counts for the entire optimizer batch, across microbatches
and data-parallel ranks. Context-parallel masks follow the same load-balancer
permutation/padding as labels. Metrics report the weighted CE, output/context CE,
token counts, and the realized context fraction of loss. Evaluation retains the
existing output-only SFT loss path.

## Validation

Run from the repo root with the olmo-core Python environment:

```bash
PYTHONPATH=src python -m pytest -q src/test/leaky_sft
shasum -a 256 -c src/test/leaky_sft/tests.sha256
```

The unchanged suite passes **43 tests** locally. **9 real CUDA/Liger cases are
skipped**, because the local machine has no CUDA device. Tests cover endpoint and
intermediate losses AND gradients, empty groups, masking, collator/loader packing,
accumulation, uneven partition arithmetic, context-parallel mask routing, config
validation, and real CPU transformer/train-module integration.

This does not establish CUDA, actual multi-rank CP/DDP/FSDP, torch.compile,
throughput, or downstream task quality. The CP test exercises routing with a
noncontiguous test load balancer; it does not run distributed attention. The
partition test checks normalization algebra, not collective communication.

Initial scope: standard LM head, dense/landmark data paths, no tensor/pipeline
parallelism or z-loss. Unsupported configurations fail explicitly. No training,
prep, or evaluation jobs were launched. Future Beaker jobs require min_runtime=1h.

Existing regression checks passed 72 tests (66 in the main run, 6 remaining tests
run separately). The existing `data_loader_test.py::test_data_loader[2]` stalled
in the local multiprocessing/selector path, including in isolation and with
OMP/MKL thread counts set to one. Those runs were interrupted; that test is
**unverified**, not counted as passed. Evidence is in `evidence/regression*.txt`
and `evidence/multiworker-interrupted.txt`. The single-worker loader tests and the
frozen LeakySFT loader tests pass. Lint and `git diff --check` also passed.
