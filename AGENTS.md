# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

BacBench is a Python package for benchmarking machine learning models on bacterial genomics tasks. The main package lives in `bacbench/`, with task-specific code under `bacbench/tasks/` and tests under `tests/`.

Key areas:

- `bacbench/modeling/`: embedding utilities and model wrappers for DNA/protein sequence models.
- `bacbench/pp/`: preprocessing utilities for genome annotations and sequence extraction.
- `bacbench/tasks/`: task implementations for essential genes, operons, PPI, strain clustering, antibiotic resistance, and phenotypic traits.
- `tests/`: pytest-based tests. Coverage is currently concentrated around modeling and preprocessing.

## Environment

This project targets Python 3.10+ and uses packaging metadata from `pyproject.toml`.

Common setup commands:

```bash
pip install -e ".[test]"
```

Preferred local setup with `uv`:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ".[test]"
```

Use the repo-local environment explicitly when running commands from an agent session:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/tasks/phenotypic_traits/test_train_and_predict_linear.py
```

Install development tools into the same environment when needed:

```bash
uv pip install ruff
.venv/bin/ruff check bacbench tests
```

Optional extras:

```bash
pip install -e ".[faesm]"
pip install -e ".[pyrodigal]"
uv pip install -e ".[faesm]"
uv pip install -e ".[pyrodigal]"
```

Only install GPU-specific extras such as `faesm` on machines with a compatible CUDA/GPU setup.

## Useful Commands

Run the test suite:

```bash
pytest
```

Run a focused test file:

```bash
pytest tests/pp/preprocess.py
```

Run linting and formatting checks with Ruff if available:

```bash
ruff check bacbench tests
ruff format --check bacbench tests
```

Format code:

```bash
ruff format bacbench tests
```

## Coding Conventions

- Prefer small, focused changes that preserve existing public APIs unless the task explicitly requires an API change.
- Keep imports sorted and formatted according to the Ruff configuration in `pyproject.toml`.
- Use type hints for new functions when practical, especially for reusable utilities.
- Follow the existing task script style: command-line entry points should be readable, explicit, and easy to run from the repo root.
- Avoid broad exception handling unless the surrounding code already expects best-effort behavior. If catching a broad exception is needed, include enough context in the error message.
- Do not introduce heavyweight dependencies without a clear reason.
- Modeling code should separate model loading from data transformation where possible so tests can cover transformation logic without loading large models.


## Data And Runtime Notes

- Many BacBench workflows use large Hugging Face datasets. Prefer streaming examples or small fixtures in tests.
- Do not add large generated files, model checkpoints, downloaded datasets, `.parquet` outputs, or local cache artifacts to the repo.
- Be careful with scripts that download data, run GPU inference, or write large outputs. Document the expected input/output paths and make dry-run or small-sample operation possible when adding new code.
- Existing `.DS_Store`, `__pycache__`, and other local artifacts should not be expanded or relied on.

## Testing Guidance

- Add or update tests for changes to shared utilities, preprocessing behavior, and model/task data readers.
- For task scripts that are difficult to test end-to-end, factor pure parsing or transformation logic into testable helpers.
- Use small in-memory fixtures rather than network downloads in unit tests.
- If a change depends on optional packages or GPU hardware, keep the default test path CPU-only and skip optional tests clearly.

## Before Finishing

When making code changes, report:

- What files changed.
- Which tests or checks were run.
- Any tests/checks that could not be run and why.
- Any assumptions about local data, optional dependencies, or hardware.
