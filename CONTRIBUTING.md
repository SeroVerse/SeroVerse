# Contributing to `SeroVerse`

Contributions are welcome across the entire **SeroVerse** ecosystem (`SeroVerse`, `SeroRef`, `KBank`, `SeroGen`, and `SeroEpi`)! Every improvement — from bug reports and documentation refinements to Numba kernels, chemical graph parsers, and Bayesian models — helps advance open bacterial genomics, glycomics, and public-health surveillance.

## Types of Contributions

### Report Bugs

Report bugs for the core engine, clients, or ecosystem documentation at https://github.com/SeroVerse/SeroVerse/issues, or on the respective pillar repository (`SeroRef`, `KBank`, `SeroGen`, `SeroEpi`).

If you are reporting a bug, please include:

- Your operating system name, Python version (`>=3.13`), and relevant library versions (`numpy`, `numba`, `scipy`).
- Minimal inputs or code snippets sufficient to reproduce the issue.
- Expected biological or algorithmic behavior vs. actual output.

### Fix Bugs & Implement Features

Look through the GitHub issues for open bugs or enhancements at https://github.com/SeroVerse/SeroVerse/issues. If you are proposing a new feature:

- Explain how it fits into `SeroVerse`'s foundational architecture (`seroverse.core` and `seroverse.client`) or downstream pillar packages (`SeroRef`, `KBank`, `SeroGen`, `SeroEpi`).
- Keep hot-path data structures in contiguous Structure-of-Arrays (SoA) NumPy buffers (`BatchedContainer` / `RaggedArrayContainer`) so Numba kernels remain zero-copy and GIL-free.
- Preserve strict module boundaries (keep `__init__.py` files empty and use explicit submodule imports).

## Local Development Setup

`SeroVerse` uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

1. Fork and clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/SeroVerse.git
cd SeroVerse
```

2. Sync all development, test, and documentation dependency groups:

```bash
uv sync --all-groups
```

3. Create a feature branch for your changes:

```bash
git checkout -b feat/your-feature-name
```

## Quality Checks & Testing

Before opening a pull request, verify that formatting, linting, static type checking, unit tests, and strict documentation builds all pass:

```bash
# 1. Lint and check formatting with Ruff
uv run ruff check .
uv run ruff format --check .

# 2. Run static type checking with ty
uv run ty check src/

# 3. Run the pytest test suite
uv run pytest

# 4. Synchronize theme changes across sibling repositories (if editing docs/overrides or docs/stylesheets)
uv run python scripts/sync_theme.py

# 5. Build the Zensical documentation site in strict mode
uv run zensical build --clean --strict
```

## Pull Request Guidelines

1. Include unit tests in `tests/` covering any new algorithms, edge cases, or bug fixes.
2. Document public classes, methods, and dataclasses using Google-style docstrings so `mkdocstrings` renders complete API references.
3. Ensure all CI checks (`ci.yml` and `docs.yml`) pass cleanly.
