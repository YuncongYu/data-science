# Data Science

A collection of alorithms, utilities, and examples that demonstrate common data-science workflows and algorithms.
The repository is organized so you can browse topics (notebooks), run small scripts, and run tests for the utilities.

## Quick Links

- Notebooks: `experiments/`
- Utilities: `data_science/util/`
- Tests: `tests/`

## Installation

This project uses `uv` and the checked-in `uv.lock` to manage dependencies.
With `uv` installed, run this from the repository root:

```bash
uv sync --locked
```

This creates the project environment and includes the `dev` dependency group
by default. For project dependencies only, run `uv sync --locked --no-dev`.

## Usage

- Open notebooks in `experiments/` with `uv run jupyter notebook`.
- Run small utilities from the project root, for example:

```bash
uv run python -m data_science.util.print_images
```

## Developer setup

After syncing the environment, run the development tools with `uv run`:

```bash
uv run ruff check .
uv run mypy data_science
uv run pytest -q
```

### Pre-commit hooks

Install the hooks configured in `.pre-commit-config.yaml` and run them against
all files:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Project Structure

- `data_science/` — package source code and utilities
- `experiments/` — notebooks and experiment folders
- `data/` — example data (gitignored if large)
- `tests/` — unit tests

## Contributing

Contributions are welcome. Suggested next steps:

- Add a `requirements.txt` or `pyproject.toml` extras for reproducible installs
- Add small runnable examples under `scripts/` or `examples/`

## Next steps / Suggestions

- Add CI (GitHub Actions) to run `pytest` and linting
- Add badges for build status and Python support
- Expand examples with reproducible datasets and commands

---
Updated README to provide clearer setup and usage instructions.
