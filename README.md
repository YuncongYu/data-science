# Data Science

A collection of alorithms, utilities, and examples that demonstrate common data-science workflows and algorithms.
The repository is organized so you can browse topics (notebooks), run small scripts, and run tests for the utilities.

## Quick Links

- Notebooks: `experiments/`
- Utilities: `data_science/util/`
- Tests: `tests/`

## Installation

This project uses `pixi` to manage the development environment. Install `pixi`
following the instructions at https://prefix.dev/tools/pixi, then install
dependencies with:

```bash
pixi install
```

For developers, use instead

```bash
pixi install --dev
```


If you prefer a traditional Python venv, create and activate a venv and then
install dependencies (if a requirements file is added later):

```bash
python -m venv .venv
source .venv/bin/activate
# pip install -r requirements.txt
```

## Usage

- Open notebooks in `experiments/` with Jupyter notebook or JupyterLab.
- Run small utilities from the project root, for example:

```bash
python -m data_science.util.print_images
```

- Run tests with `pytest` (install it into your environment first):

```bash
pytest -q
```

## Developer setup

Follow these steps to prepare a development environment and install the dev
tooling used by the project.

1) Create / activate an environment

- Using `pixi` (recommended for reproducible environments):

	```bash
	pixi install
	# use `pixi run <cmd>` to run commands inside the pixi-managed environment
	```

- Using a Python `venv`:

	```bash
	python -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	```

2) Install developer tools (preferred: pixi)

Prefer adding development tools to your `pixi` environment so they are
installed and managed by `pixi` rather than calling `pip` directly. Consult
the pixi documentation for the exact manifest fields; after adding the dev
dependencies run:

```bash
pixi install
```

Run the developer tools inside the pixi-managed environment using `pixi run`:

```bash
pixi run ruff check .
pixi run mypy data_science
pixi run pytest -q
```

If you must work without `pixi`, use a Python virtual environment and install
the tools locally (not recommended if you want reproducible team environments):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -U ruff mypy pytest
```

3) Common developer commands

- Lint the codebase: `pixi run ruff check .` (or `ruff check .` inside a venv)
- Run static type checks: `pixi run mypy data_science`
- Run the test-suite: `pixi run pytest -q`

Add these commands to CI so they run automatically on pull requests.

### Pre-commit hooks

This repository includes a Git pre-commit configuration at
`.pre-commit-config.yaml`. We recommend installing `pre-commit` as a dev
dependency in your `pixi` manifest and running the hook installer from the
pixi-managed environment:

```bash
pixi run pre-commit install
```

To run the configured hooks against all files (useful for CI or one-off checks):

```bash
pixi run pre-commit run --all-files
```

If you cannot use `pixi`, install `pre-commit` in your local venv and run the
same commands inside that environment. Adding the `pre-commit` step to CI is a
good way to ensure consistent formatting and checks on pull requests.

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
