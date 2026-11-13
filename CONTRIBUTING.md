# Contribute to Olmo-core

Thank you for your interest in contributing to Olmo-core! This guide will help you get set up, understand the layout of the repository, implement a change, validate it, and submit a pull request (PR).

## Setting up your development environment

1. For rapid experimentation we recommend forking Olmo-core for your project instead of installing it as a dependency. Start by [creating a fork](https://github.com/allenai/Olmo-core/fork) if you haven't already, and then cloning your fork to the computer where you'll be doing the development.
2. Create or activate a Python virtual environment with Python 3.10 or newer. We recommend [uv](https://docs.astral.sh/uv/): `uv venv --python 3.11 && source .venv/bin/activate`. Any other virtual environment manager (such as `python -m venv` or conda) works as well.
3. Once you've `cd`-ed into the root directory of your clone of Olmo-core *and* activated your virtual environment, install [PyTorch](https://pytorch.org) according to the directions specific to your operating system and hardware (a CPU-only distribution is fine for local development). Finally, install Olmo-core in editable mode by running:

    ```bash
    pip install -e '.[all]'
    ```

    or, if using `uv`:

    ```bash
    uv sync --all-extras
    ```

If you plan to build the documentation locally you'll also need Sphinx: both commands above install the `docs` extra automatically because it is part of `all`.

## Codebase tour

- `src/olmo_core/` contains the library code. Major sub-packages include:
  - `data/` for dataset abstractions and loaders used during training.
  - `distributed/` for launch utilities and collective communication helpers.
  - `nn/` for neural network building blocks (attention, rotary embeddings, transformer modules, etc.).
  - `optim/`, `ops/`, and `kernels/` for optimizer, custom op, and GPU kernel implementations.
  - `train/` and `launch/` for end-to-end training orchestration utilities.
- `src/examples/` holds runnable examples and reference scripts.
- `src/scripts/` contains CLI tools for training, evaluation, releases, and infrastructure workflows.
- `src/test/` mirrors the library layout and contains the unit and integration tests.
- `docs/` is the Sphinx documentation tree. API docs are generated from docstrings inside `src/olmo_core`.

The best way to find relevant code is to start in `src/olmo_core` and look for a matching module name. Tests in `src/test` link back to the features they cover and are a good reference when adding new ones.

## Making a change

1. Start by opening or finding an issue describing the problem you are solving. If none exists, consider filing one so maintainers are aware of the context.
2. Create a feature branch from `main` in your fork and make your changes.
3. Update or add docstrings when you introduce new functionality. Docstrings are automatically incorporated into Olmo-core's [documentation](https://olmo-core.readthedocs.io/en/latest/overview/introduction.html).
4. Add or update tests in `src/test` to cover your changes. Favor fast, deterministic tests that can run on CPU-only environments whenever possible.
5. Update`CHANGELOG.md` with a quick description of your change.

## Running tests

We use `pytest` for the test suite. To run all tests locally:

```bash
pytest -v src/test
```

You can target a subset of tests by passing a file path (for example `pytest src/test/nn/rope_test.py`) or a keyword expression (for example `pytest -k rope`). All tests should pass before you open a PR. Some tests exercise GPU-specific code paths; they automatically skip themselves if the required hardware is unavailable.


### Test conventions

- Place new test modules under `src/test` and name them `*_test.py` so that `pytest` discovers them automatically. Mirror the package structure of the code you are testing when it makes sense (e.g., tests for `src/olmo_core/nn/rope.py` live in `src/test/nn/rope_test.py`).
- Name individual test functions `test_*` and prefer `pytest.mark.parametrize` to cover multiple inputs or configurations without duplicating code. Parametrized tests keep runtime manageable while exercising the variations Olmo-core supports.

### CPU, GPU, and multi-GPU runs

- GPU-only scenarios use the `gpu` marker applied by helpers such as `@requires_gpu`. These tests skip automatically when CUDA is unavailable. To focus on them explicitly run `pytest -m gpu`; to exclude them on CPU-only machines run `pytest -m "not gpu"`.
- Multi-GPU suites use `@requires_multi_gpu` and the `run_distributed_test(...)` helper to launch distributed workers. Reserve at least two visible devices before running them, for example `CUDA_VISIBLE_DEVICES=0,1 pytest -m gpu src/test/nn/transformer/model_test.py`.
- Some GPU tests depend on optional libraries like `flash-attn` or `grouped_gemm`. Install the full extras (`pip install -e '.[all]'`) and confirm the libraries load if you need to exercise those paths.

## Formatting and Linting

- We use `isort` and `black` for import order and code formatting: `make style-check` verifies both. If the check fails you can run `isort .` and `black .` (or the equivalent commands in your editor) to apply fixes.
- `ruff` is the primary linter. Run `make lint-check` to ensure lint rules pass. Add `--fix` locally (`ruff check --fix .`) to automatically resolve simple issues.
- `mypy` enforces static type hints. Run `make type-check` to validate that types are correct.
- Running `make checks` will execute all of the above in one go.

## Submitting a pull request (PR)

1. Push your branch to your fork, then [create a PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) targeting `allenai/Olmo-core:main`.
2. In the PR description, summarize the change, highlight any user-facing impact, and call out areas where you would like reviewer attention.
3. Confirm that you've run the relevant checks (`pytest`, `make checks`, and documentation builds if applicable) and include any notable failures or skips. Ensure that all GitHub actions that are triggered are passing.
4. Respond to review feedback promptly and keep the PR focused. If a review uncovers additional work beyond the scope, consider creating follow-up issues.

If you have questions or need guidance, feel free to open a GitHub issue.
