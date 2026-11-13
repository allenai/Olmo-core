# AGENTS.md

This file provides guidance to agents like Claude Code (claude.ai/code) and Codex (chatgpt.com/codex) when working with code in this repository.

## Overview

Olmo-core is AI2's training library for the Open Language Model (OLMo) series. It provides modular components for transformer architectures, distributed training, data loading, and evaluation.

## Commands

```bash
# Install (development)
pip install -e '.[all]'

# Run all tests (GPU tests auto-skip without GPU)
pytest -v src/

# Run a specific test file
pytest -v src/test/path/to/test_file.py

# Filter to specific tests by keywords
pytest -v src/test/path/to/test_file.py -k 'keyword'

# Auto-format code
make style

# Check formatting, lint, and types
make checks          # all three at once
make style-check     # isort + black
make lint-check      # ruff
make type-check      # mypy
```

## Code Style

- Line length: 100
- Formatting: `isort` (profile=black) + `black`
- Linting: `ruff` (ignores F403, F405, E501; F401 ignored in `__init__.py`)
- Type checking: `mypy` with `ignore_missing_imports = true`

## Docstrings

- Docstrings should be included on all public classes, methods, and functions.
- We use Sphinx to automatically build API docs by pulling from those docstrings.
- The syntax of the docstrings is a superset of reStructuredText with additional Sphinx-specific syntax for things like:
  - Cross-document links, e.g.:
    ```
    :class:`foo.Foo`  <- links to the class named 'Foo' in the module 'foo'
    :mod:`foo`        <- links to the module named 'foo'
    :func:`foo.bar`   <- links to the function named 'bar' in the module named 'foo'
    ```
  - Documenting parameters (`:param ...:`), return values (`:returns:`), or expected exceptions (`:raises ...:`).

Here's a toy example for a function:

```python
def read_file(path: str) -> str:
    """
    Read a file from disk.

    :param path: The path to the file.

    :returns: The contents of the file.

    :raises FileNotFoundError: If the file doesn't exist.
    """
    pass
```

## Architecture

### Configuration System (`src/olmo_core/config.py`)

Everything is configured via `@dataclass` classes inheriting from `Config`. This is the central design pattern:
- Configs support YAML/JSON serialization, command-line overrides via dot notation (`--train_module.optim.lr=6e-3`), and `merge()` with dotlists.
- The `Registrable` mixin (from `dataclass-extensions`) enables polymorphic config fields — a base config class can resolve to different subclasses at runtime based on a `type` field. Used in optimizers, schedulers, attention backends, and data loaders.
- Nested configs compose modularly: `TrainerConfig` contains `CheckpointerConfig`, `OptimConfig`, etc.

### Training Pipeline (`src/olmo_core/train/`)

- `Trainer` / `TrainerConfig`: Core training loop with checkpointing, evaluation, and an extensible callback system (`callbacks/`).
- `TrainModule`: Wraps the model with forward/backward logic and optimizer. The main concrete implementation is `TransformerTrainModule` / `TransformerTrainModuleConfig`, which handles parallelism setup (DP, TP, PP, CP, EP configs all live here).

### Model Architecture (`src/olmo_core/nn/`)

- `transformer/`: Core transformer with configurable blocks. `TransformerConfig` has factory methods like `olmo2_32B()` for predefined architectures.
- `attention/`: Multi-head attention with backends (flash attention, ring attention, etc.).
- `moe/`: Mixture of Experts with expert parallelism.
- `feed_forward.py`, `layer_norm.py`, `rope.py`, `lm_head.py`: Standard components.

### Data Loading (`src/olmo_core/data/`)

- `NumpyDataset` variants: Memory-mapped numpy datasets for pre-tokenized data (`.npy` files).
- `composable/`: The preferred data loading API, built on a pipeline of `TokenSource` -> `InstanceSource` -> `ComposableDataLoader`. Sources can be sliced, sampled, mixed with ratios, and split for curriculum learning. Use `InstanceSource.visualize()` to inspect the source tree. See the module docstring in `src/olmo_core/data/composable/__init__.py` for detailed examples.
- `mixes/`: Predefined data mixture configs (dolma17, OLMoE-mix-0824, etc.) with paths to tokenized data by source and tokenizer.
- Training data is stored on AI2 infrastructure (Weka filesystem, GCS). For local development, use small validation sets or synthetic data.

### Distributed Training (`src/olmo_core/distributed/`)

- `parallel/`: Implementations of data (FSDP/HSDP/DDP), tensor, pipeline, context (ring attention), and expert parallelism. These can be combined for multi-dimensional parallelism.
- `checkpoint/`: Distributed checkpointing with various filesystem backends.

### Optimization (`src/olmo_core/optim/`)

- Optimizer configs (`AdamWConfig`, `SkipStepAdamWConfig`, `LionConfig`) and LR schedulers (`CosWithWarmup`, etc.).
- `SkipStepOptimizer`: Wrapper for gradient clipping with loss spike detection.

### Examples (src/examples)

Runnable, self-contained examples and reference scripts.

### Training Scripts

Two patterns exist:

**Official scripts** (`src/scripts/official/`): Use `ExperimentConfig` + `main()` from `src/olmo_core/script_utils.py`. Launched with `torchrun` or Beaker. These reproduce published model runs.

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints
```

**Internal scripts** (`src/scripts/train/`): Use `prepare_cli_environment()` with commands (`launch`, `train`, `train_single`, `prep`, `dry_run`). See `template.py` for the starting point.

```bash
python src/scripts/train/OLMo2-1B.py dry_run test-run ai2/titan-cirrascale
python src/scripts/train/OLMo2-1B.py launch olmo2-1b-test ai2/jupiter-cirrascale-2 --launch.num_nodes=4
```

## Docker and Beaker Launch

The Docker image (`src/Dockerfile`) is a two-stage build: a `build` stage compiles GPU-specific dependencies (flash-attn, TransformerEngine, grouped_gemm, ring-flash-attn, etc.) on an NVIDIA CUDA devel image, and a `release` stage copies the conda environment into a lighter Ubuntu base with AWS CLI, Google Cloud SDK, and MLNX OFED drivers. The image contains all dependencies but *not* the Olmo-core package itself — source code is cloned at runtime.

```bash
# Build locally (versions configured in Makefile)
make docker-image

# Push to GHCR
make ghcr-image

# Create Beaker image
make beaker-image
```

**How launch works**: When a training script uses the `launch` command (or `BeakerLaunchConfig.launch()`), it creates a Gantry recipe that:
1. Starts a container from a pre-built Beaker image (default: `OLMoCoreBeakerImage.stable` in `src/olmo_core/launch/beaker.py`)
2. Clones the git repo at the current commit into the container (requires clean working tree unless `allow_dirty=True`)
3. Installs the package from source (`pip install -e .`)
4. Runs the training command, optionally wrapped with `torchrun` for multi-GPU/multi-node

Pre-built images are listed in the `OLMoCoreBeakerImage` enum in `src/olmo_core/launch/beaker.py`, tagged by torch version and CUDA version (e.g., `tch2100cu128`). The `stable` image tracks the default torch/CUDA versions. When updating default images, also update `.github/workflows/main.yml`.

`BeakerLaunchConfig` also supports `pre_setup` and `post_setup` hooks for running commands before/after the package install step, Weka bucket mounts, and multi-node settings (replicas, leader selection, host networking).

## Testing

- Tests in `src/test/` mirror the source structure.
- Name individual test functions `test_*` and prefer `pytest.mark.parametrize` to cover multiple inputs or configurations without duplicating code.
- GPU tests use `@pytest.mark.gpu` and are skipped without a GPU.
- Distributed tests use helpers in `src/olmo_core/testing/distributed.py`.
